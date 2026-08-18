from __future__ import annotations
import torch, torch.nn as nn

def _mlp(dims, act=nn.GELU, last_act=True):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act: layers.append(act())
    return nn.Sequential(*layers)

class PointNetEncoder(nn.Module):
    def __init__(self, in_dim=6, feat_dim=128, glob_dim=256):
        super().__init__()
        self.point_mlp = _mlp([in_dim, 64, feat_dim]); self.post_mlp = _mlp([feat_dim, glob_dim])
    def forward(self, x):
        f = self.point_mlp(x); return f, self.post_mlp(f.max(dim=2).values)


##################  test add Cross Attention

class CrossFragmentAttention(nn.Module):
    """Chaque point d'un fragment attend un sous-échantillon des points des
    autres fragments. Position 3D + normale injectées dans Q/K pour un a
    priori géométrique explicite (proximité, normales qui se font face) au
    lieu de tout faire reposer sur un contenu appris avec peu de données."""
    def __init__(self, feat_dim, d_model, n_heads=4, n_sub=128):
        super().__init__()
        self.n_sub = n_sub
        self.q_proj = nn.Linear(feat_dim + 3 + 3, d_model)
        self.kv_proj = nn.Linear(feat_dim + 3 + 3, d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, pt_feat, local_pos, normals, pad_mask):
        B, K, N, F = pt_feat.shape
        feat_full = torch.cat([pt_feat, local_pos, normals], -1)

        if self.training and self.n_sub < N:
            step = N // self.n_sub
            idx = torch.arange(0, N, step, device=pt_feat.device)[:self.n_sub]
            sub = feat_full[:, :, idx, :]     # sous-échantillon FIXE, pas aléatoire
        else:
            sub = feat_full
        S = sub.shape[2]

        outs = []
        for k in range(K):
            others = torch.cat([sub[:, j] for j in range(K) if j != k], dim=1)
            q = self.q_proj(feat_full[:, k])
            kv = self.kv_proj(others)
            other_pad = torch.cat([pad_mask[:, j:j+1].expand(-1, S)
                                   for j in range(K) if j != k], dim=1)
            o, _ = self.mha(q, kv, kv, key_padding_mask=other_pad)
            outs.append(o)
        return torch.stack(outs, dim=1)

##################

class FragmentReduceNet(nn.Module):
    def __init__(self, feat_dim=128, glob_dim=256, d_model=256, n_heads=4, n_layers=4,
                 dropout=0.0, use_normals=True):
        super().__init__()
        self.use_normals = use_normals
        self.encoder = PointNetEncoder(6 if use_normals else 3, feat_dim, glob_dim)
        self.token_proj = _mlp([glob_dim + 3 + 3, d_model])
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.decoder = _mlp([feat_dim + d_model + 3, 256, 128, 3], last_act=False)
        ########
        # self.cross_attn = CrossFragmentAttention(feat_dim, feat_dim, n_heads=4, n_sub=256)
        # self.decoder = _mlp([feat_dim + d_model + 3 + feat_dim, 256, 128, 3], last_act=False)
        # #                                                ^^^^^^^^ nouveau terme, dimension du cross_feat
        ##########
    def forward(self, pts_norm, normals, centroids, bone, pad_mask=None):
        B, K, N, _ = pts_norm.shape
        local = pts_norm - centroids.unsqueeze(2)
        x = torch.cat([local, normals], -1) if self.use_normals else local
        pt_feat, glob = self.encoder(x)
        tok = self.token_proj(torch.cat([glob, bone, centroids], -1))
        ctx = self.transformer(tok, src_key_padding_mask=pad_mask)
        ctx_exp = ctx.unsqueeze(2).expand(-1, -1, N, -1)

        # #######
        # pad = pad_mask if pad_mask is not None else torch.zeros(B, K, dtype=torch.bool, device=pts_norm.device)
        # cross_feat = self.cross_attn(pt_feat, local, normals, pad)
        # return self.decoder(torch.cat([pt_feat, ctx_exp, local, cross_feat], -1))
        # ##########
        return self.decoder(torch.cat([pt_feat, ctx_exp, local], -1))

def masked_mse(pred, target, pad_mask):
    valid = (~pad_mask).float().unsqueeze(-1).unsqueeze(-1)
    diff = ((pred - target) ** 2) * valid
    return diff.sum() / (valid.sum() * pred.shape[2] * pred.shape[3]).clamp(min=1.0)
