"""Shared I/O + geometry utilities (torch-free) for PENGWIN 2026 Task 3."""
from __future__ import annotations
import json, glob, os
import numpy as np
import trimesh


def bone_of(fid: int):
    if 1 <= fid <= 100: return "SA"
    if 101 <= fid <= 200: return "LI"
    if 201 <= fid <= 300: return "RI"
    return None

BONE_TO_IDX = {"SA": 0, "LI": 1, "RI": 2}

def bone_onehot(fid: int) -> np.ndarray:
    v = np.zeros(3, dtype=np.float32); b = bone_of(fid)
    if b is not None: v[BONE_TO_IDX[b]] = 1.0
    return v

def load_obj_meshes(obj_path: str) -> dict:
    scene = trimesh.load(obj_path, split_object=True, process=False)
    meshes = {}
    if isinstance(scene, trimesh.Scene):
        for key, geom in scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh): meshes[str(key)] = geom
    elif isinstance(scene, trimesh.Trimesh):
        meshes["1"] = scene
    return meshes

def find_input_obj(input_dir: str = "/input") -> str:
    for c in ("peripelvic-fracture-fragments-meshes.obj", "peripelvic-fracture-fragments.obj"):
        p = os.path.join(input_dir, c)
        if os.path.isfile(p): return p
    g = sorted(glob.glob(os.path.join(input_dir, "**", "*.obj"), recursive=True))
    if g: return g[0]
    raise FileNotFoundError(f"No .obj under {input_dir}")

def sample_points(mesh, n, rng):
    if len(mesh.vertices) == 0: return np.zeros((n, 3))
    if len(mesh.faces) > 0:
        pts, _ = trimesh.sample.sample_surface(mesh, n); return np.asarray(pts, float)
    idx = rng.integers(0, len(mesh.vertices), size=n); return np.asarray(mesh.vertices[idx], float)

def apply_T(pts, T):
    pts = np.asarray(pts, float); ones = np.ones((len(pts), 1))
    return (T @ np.concatenate([pts, ones], 1).T).T[:, :3]

def kabsch(P, Q):
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cQ - R @ cP

def make_T(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T

def anchor_id(fragment_ids):
    ids = [str(f) for f in fragment_ids]
    sa = sorted([f for f in ids if bone_of(int(f)) == "SA"], key=lambda x: int(x))
    return sa[0] if sa else sorted(ids, key=lambda x: int(x))[0]

def anchor_normalise(poses):
    a = anchor_id(poses.keys()); Tinv = np.linalg.inv(poses[a])
    return {fid: Tinv @ T for fid, T in poses.items()}

def write_poses_json(poses, out_path):
    entries = [{"fragment_id": str(fid), "transformation": np.asarray(poses[fid], float).tolist()}
               for fid in sorted(poses.keys(), key=lambda x: int(x))]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f: json.dump(entries, f, indent=2)

def read_poses_json(path):
    with open(path) as f: data = json.load(f)
    if isinstance(data, list):
        return {str(e["fragment_id"]): np.array(e["transformation"], float) for e in data}
    return {str(k): np.array(v, float) for k, v in data.items()}

def identity_poses(fragment_ids):
    return {str(f): np.eye(4) for f in fragment_ids}
