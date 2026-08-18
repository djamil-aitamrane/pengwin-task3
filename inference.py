"""PENGWIN 2026 Task 3 container entry point. Reads /input, writes /output.

Always emits one pose per fragment, anchor (first SA fragment) forced to
identity, and falls back to identity on any failure (never crashes).
"""
from __future__ import annotations
import os, sys, glob, argparse, traceback
import numpy as np
from data_io import (load_obj_meshes, find_input_obj, kabsch, make_T,
                     anchor_normalise, write_poses_json, identity_poses)


def find_checkpoint(model_dir):
    if model_dir and os.path.isfile(model_dir): return model_dir
    for d in ([model_dir] if model_dir else []) + ["/opt/ml/model", "./model"]:
        if d and os.path.isdir(d):
            c = sorted(glob.glob(os.path.join(d, "*.pt")))
            c.sort(key=lambda p: ("_last" in os.path.basename(p), p))
            if c: return c[0]
    return None


def predict_with_model(meshes, ckpt_path):
    import torch
    from model_test import FragmentReduceNet
    from preprocess import build_features
    ckpt = torch.load(ckpt_path, map_location="cpu"); cfg = ckpt.get("cfg", {})
    n_points = int(cfg.get("n_points", 512)); use_normals = bool(cfg.get("use_normals", True))
    model = FragmentReduceNet(feat_dim=cfg.get("feat_dim", 128), glob_dim=cfg.get("glob_dim", 256),
                              d_model=cfg.get("d_model", 256), n_heads=cfg.get("n_heads", 4),
                              n_layers=cfg.get("n_layers", 4), dropout=0.0, use_normals=use_normals)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    rng = np.random.default_rng(42)
    feat = build_features(meshes, n_points, rng)
    pts = torch.from_numpy(feat["pts_norm"]).unsqueeze(0)
    nrm = torch.from_numpy(feat["normals"]).unsqueeze(0)
    cen = torch.from_numpy(feat["centroids"]).unsqueeze(0)
    bone = torch.from_numpy(feat["bone"]).unsqueeze(0)
    with torch.no_grad():
        pred = model(pts, nrm, cen, bone, pad_mask=None)[0].numpy()
    pred_mm = pred * feat["scale"] + feat["center"]
    poses = {}
    for k, fid in enumerate(feat["ids"]):
        R, t = kabsch(feat["raw"][k], pred_mm[k]); poses[str(fid)] = make_T(R, t)
    return poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="/input")
    ap.add_argument("--output", default="/output/reduction-poses-matrices.json")
    ap.add_argument("--model", default="/opt/ml/model")
    args = ap.parse_args()

    obj_path = find_input_obj(args.input_dir)
    meshes = load_obj_meshes(obj_path)
    frag_ids = list(meshes.keys())
    print(f"[inference] {obj_path} -> {len(frag_ids)} fragments", flush=True)

    poses = None
    ckpt = find_checkpoint(args.model)
    if ckpt is not None:
        try:
            poses = predict_with_model(meshes, ckpt)
            print(f"[inference] predicted with {ckpt}", flush=True)
        except Exception:
            print("[inference] model failed -> identity:", flush=True); traceback.print_exc(); poses = None
    else:
        print("[inference] no checkpoint -> identity baseline", flush=True)

    if poses is None:
        poses = identity_poses(frag_ids)
    for fid in frag_ids:
        if str(fid) not in poses: poses[str(fid)] = np.eye(4)
    try:
        poses = anchor_normalise(poses)
    except Exception:
        traceback.print_exc()
    write_poses_json(poses, args.output)
    print(f"[inference] wrote {args.output} ({len(poses)} fragments)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        try:
            in_dir, out = "/input", "/output/reduction-poses-matrices.json"
            for i, a in enumerate(sys.argv):
                if a == "--input_dir" and i + 1 < len(sys.argv): in_dir = sys.argv[i + 1]
                if a == "--output" and i + 1 < len(sys.argv): out = sys.argv[i + 1]
            meshes = load_obj_meshes(find_input_obj(in_dir))
            write_poses_json(anchor_normalise(identity_poses(list(meshes.keys()))), out)
            print("[inference] emergency identity output", flush=True)
        except Exception:
            traceback.print_exc(); sys.exit(1)
