"""model.py — PointReduceNet : tokens au niveau point, encodage de Fourier,
attention hiérarchique intra-fragment puis inter-fragments.

Sortie (B, K, N, 3) : coordonnées assemblées prédites, comme la version
précédente. losses.py, train.py, dataset.py et inference.py sont inchangés.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    """Encodage sinusoïdal multi-échelle : (...,3) -> (...,3*(1+2*multires))."""
    def __init__(self, multires=6):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(multires) * math.pi)
        self.dim = 3 * (1 + 2 * multires)

    def forward(self, x):
        xf = x.unsqueeze(-1) * self.freqs
        return torch.cat([x, torch.cat([xf.sin(), xf.cos()], -1).flatten(-2)], -1)


class MHA(nn.Module):
    """Attention multi-tête sans matérialisation de la matrice d'attention."""
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.h, self.dk, self.p = n_heads, d_model // n_heads, dropout
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def _split(self, x):                                   # (B,L,d) -> (B,h,L,dk)
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dk).transpose(1, 2)

    def forward(self, q, kv, attn_mask=None):
        B, Lq, d = q.shape
        out = F.scaled_dot_product_attention(
            self._split(self.q(q)), self._split(self.k(kv)), self._split(self.v(kv)),
            attn_mask=attn_mask, dropout_p=self.p if self.training else 0.0)
        return self.o(out.transpose(1, 2).reshape(B, Lq, d))


class HierarchicalBlock(nn.Module):
    """Attention intra-fragment, puis inter-fragments, puis feed-forward."""
    def __init__(self, d_model, n_heads, n_sub, dropout=0.0):
        super().__init__()
        self.n_sub = n_sub
        self.intra = MHA(d_model, n_heads, dropout)
        self.inter = MHA(d_model, n_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                nn.Linear(2 * d_model, d_model))
        self.n1, self.n2, self.n3 = (nn.LayerNorm(d_model) for _ in range(3))

    def forward(self, x, pad_mask):                        # x (B,K,N,d), pad_mask (B,K)
        B, K, N, d = x.shape

        # intra-fragment : chaque fragment est une séquence indépendante.
        # Pas de masque, les fragments paddés sont écartés plus tard par la perte.
        h = self.n1(x).reshape(B * K, N, d)
        x = x + self.intra(h, h).reshape(B, K, N, d)

        # inter-fragments : requêtes = tous les points, clés = sous-échantillon global.
        # attn_mask suit la convention de SDPA, True = participe à l'attention.
        h = self.n2(x)
        idx = torch.linspace(0, N - 1, min(self.n_sub, N), device=x.device).long()
        kv = h[:, :, idx, :].reshape(B, -1, d)
        keep = (~pad_mask).repeat_interleave(len(idx), dim=1).view(B, 1, 1, -1)
        x = x + self.inter(h.reshape(B, K * N, d), kv, attn_mask=keep).reshape(B, K, N, d)

        return x + self.ff(self.n3(x))


class PointReduceNet(nn.Module):
    """`feat_dim` et `glob_dim` sont acceptés pour compatibilité et ignorés."""
    def __init__(self, feat_dim=128, glob_dim=256, d_model=256, n_heads=4, n_layers=4,
                 dropout=0.0, use_normals=True, n_sub=64, multires=6, max_frags=64):
        super().__init__()
        self.use_normals = use_normals
        self.max_frags = max_frags
        self.fourier = FourierFeatures(multires)
        f = self.fourier.dim
        in_dim = 2 * f + (f if use_normals else 0) + 32
        self.frag_embed = nn.Embedding(max_frags, 16)
        self.bone_proj = nn.Linear(3, 16)
        self.token_proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.GELU(),
                                        nn.LayerNorm(d_model))
        self.blocks = nn.ModuleList([HierarchicalBlock(d_model, n_heads, n_sub, dropout)
                                     for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, 3))

    def forward(self, pts_norm, normals, centroids, bone, pad_mask=None):
        B, K, N, _ = pts_norm.shape
        if pad_mask is None:
            pad_mask = torch.zeros(B, K, dtype=torch.bool, device=pts_norm.device)
        local = pts_norm - centroids.unsqueeze(2)
        fid = torch.arange(K, device=pts_norm.device).clamp(max=self.max_frags - 1)
        fid = fid.view(1, K, 1).expand(B, K, N)

        parts = [self.fourier(pts_norm),                   # position globale
                 self.fourier(local)]                      # forme relative au centroïde
        if self.use_normals:
            parts.append(self.fourier(normals))            # orientation locale
        parts += [self.frag_embed(fid),                    # identité de pièce apprise
                  self.bone_proj(bone).unsqueeze(2).expand(-1, -1, N, -1)]   # type d'os

        x = self.token_proj(torch.cat(parts, -1))
        for blk in self.blocks:
            x = blk(x, pad_mask)
        return self.head(x)


# Alias de compatibilité : train.py et inference.py instancient ce nom.
FragmentReduceNet = PointReduceNet


def masked_mse(pred, target, pad_mask):
    valid = (~pad_mask).float().unsqueeze(-1).unsqueeze(-1)
    diff = ((pred - target) ** 2) * valid
    return diff.sum() / (valid.sum() * pred.shape[2] * pred.shape[3]).clamp(min=1.0)