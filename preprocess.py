from __future__ import annotations
import numpy as np
import trimesh
from data_io import bone_onehot, apply_T

BONE_SWAP = {"SA": "SA", "LI": "RI", "RI": "LI"}


def subsample_idx(m, n, rng):
    if m == 0: return np.zeros(n, int)
    return rng.choice(m, size=n, replace=m < n)


def sample_points_normals(mesh, n, rng):
    if len(mesh.faces) > 0:
        pts, fidx = trimesh.sample.sample_surface(mesh, n)
        return np.asarray(pts, np.float32), np.asarray(mesh.face_normals[fidx], np.float32)
    v = np.asarray(mesh.vertices, np.float32)
    idx = rng.integers(0, len(v), size=n)
    return v[idx], np.zeros((n, 3), np.float32)


def normalize_points(raw_pts):
    """raw_pts: (K,N,3). Centre on mean, scale by radius. Normals NOT touched."""
    allp = raw_pts.reshape(-1, 3); center = allp.mean(0)
    scale = float(np.max(np.linalg.norm(allp - center, axis=1))) or 1.0
    pn = (raw_pts - center) / scale
    return pn.astype(np.float32), pn.mean(1).astype(np.float32), center, scale


def _rand_rot(rng, max_deg):
    axis = rng.normal(size=3); axis /= (np.linalg.norm(axis) + 1e-9)
    a = np.deg2rad(rng.uniform(0, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


# ---------------------------------------------------------------------------
# Clinical / inference: features from meshes (with normals)
# ---------------------------------------------------------------------------
def build_features(meshes, n_points, rng):
    ids = sorted(meshes.keys(), key=lambda x: int(x))
    pts, nrm = [], []
    for i in ids:
        p, n = sample_points_normals(meshes[i], n_points, rng)
        pts.append(p); nrm.append(n)
    raw = np.stack(pts, 0); normals = np.stack(nrm, 0)
    pts_norm, centroids, center, scale = normalize_points(raw)
    bone = np.stack([bone_onehot(int(i)) for i in ids], 0).astype(np.float32)
    return {"ids": ids, "raw": raw, "normals": normals.astype(np.float32),
            "center": center, "scale": scale, "pts_norm": pts_norm,
            "centroids": centroids, "bone": bone}


def target_coords(feat, gt_poses):
    c, s = feat["center"], feat["scale"]
    return np.stack([(apply_T(feat["raw"][k], gt_poses[str(fid)]) - c) / s
                     for k, fid in enumerate(feat["ids"])], 0).astype(np.float32)


def make_training_sample(frags, n_points, rng, max_rot_deg=30.0, max_trans_mm=25.0,
                         global_rot=True, flip_prob=0.5, jitter_std=0.0):
    bones = [f["bone"] for f in frags]
    Rg = _rand_rot(rng, 180.0) if global_rot else np.eye(3)
    flip = rng.random() < flip_prob
    red_p, red_n = [], []
    for f in frags:
        p = f["points"].astype(np.float64); n = f["normals"].astype(np.float64)
        if flip:
            p = p * np.array([-1.0, 1.0, 1.0]); n = n * np.array([-1.0, 1.0, 1.0])
        red_p.append((Rg @ p.T).T); red_n.append((Rg @ n.T).T)
    if flip:
        bones = [BONE_SWAP[b] for b in bones]
    order = sorted(range(len(frags)), key=lambda k: (bones[k] != "SA", k))
    anchor = order[0]
    K = len(frags)
    rp = np.zeros((K, n_points, 3)); rn = np.zeros((K, n_points, 3))
    ip = np.zeros((K, n_points, 3)); inn = np.zeros((K, n_points, 3))
    for k in range(K):
        idx = subsample_idx(len(red_p[k]), n_points, rng)
        rp[k] = red_p[k][idx]; rn[k] = red_n[k][idx]
        if k == anchor:
            ip[k] = rp[k]; inn[k] = rn[k]
        else:
            R = _rand_rot(rng, max_rot_deg); t = rng.uniform(-max_trans_mm, max_trans_mm, 3)
            ip[k] = (R @ rp[k].T).T + t; inn[k] = (R @ rn[k].T).T
    if jitter_std > 0:
        ip = ip + rng.normal(0, jitter_std, ip.shape)
    pts_norm, centroids, center, scale = normalize_points(ip)
    target = ((rp - center) / scale).astype(np.float32)
    gid = {"SA": 1, "LI": 101, "RI": 201}
    bone = np.stack([bone_onehot(gid[b]) for b in bones], 0).astype(np.float32)
    return {"pts_norm": pts_norm, "normals": inn.astype(np.float32),
            "centroids": centroids, "bone": bone, "target": target}
