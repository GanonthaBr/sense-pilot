"""
generate.py — Run TerraMind S2L2A -> S1GRD on N paired TerraMesh tiles and save
everything evaluate.py needs as .npz files (one per tile).

Saves per tile: real S2 (12,224,224), real S1 (2,224,224) dB, generated S1
(2,224,224), DEM (224,224), TWI (224,224).

Usage (login node streams data; GPU node must have data/model cached — see README):
    python src/generate.py --config configs/pilot.yaml [--device cpu|cuda] [--n 50]
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from data import iter_paired_samples, compute_twi, HF_PATH


def force_to_device(module: torch.nn.Module, device: str) -> None:
    """Move ALL tensor attributes to device, including unregistered buffers.
    Fixes TerraMind's pretraining_mean/std living on CPU after .to(device)."""
    for name, attr in list(vars(module).items()):
        if torch.is_tensor(attr):
            setattr(module, name, attr.to(device))
    for child in module.children():
        force_to_device(child, device)


def build_model(cfg: dict, device: str):
    from terratorch import FULL_MODEL_REGISTRY
    m = FULL_MODEL_REGISTRY.build(
        cfg["model"]["name"],
        pretrained=True,
        modalities=cfg["model"]["input_modalities"],
        output_modalities=cfg["model"]["output_modalities"],
        timesteps=cfg["model"]["timesteps"],
        standardize=cfg["model"]["standardize"],
    )
    m = m.eval().to(device)
    force_to_device(m, device)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--device", default=None, help="override config device (cpu/cuda)")
    ap.add_argument("--n", type=int, default=None, help="override number of tiles")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = args.device or cfg["model"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: cuda requested but unavailable -> falling back to cpu")
        device = "cpu"
    n_tiles = args.n or cfg["data"]["n_tiles"]

    out_dir = Path("outputs/tiles")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TerraMind on {device}...")
    model = build_model(cfg, device)

    print(f"Streaming paired samples (need {n_tiles} with S1GRD)...")
    done = 0
    for s in iter_paired_samples(path=cfg["data"].get("hf_path", HF_PATH), max_samples=None):
        out_path = out_dir / f"{s['key']}.npz"
        if out_path.exists():  # resumable
            done += 1
            if done >= n_tiles:
                break
            continue

        x = torch.from_numpy(s["s2"]).float().unsqueeze(0).to(device)  # (1,12,224,224)
        with torch.no_grad():
            out = model({"S2L2A": x})
        gen_s1 = out["S1GRD"].squeeze(0).cpu().numpy()  # (2,224,224)

        twi = compute_twi(s["dem"])

        np.savez_compressed(
            out_path,
            s2=s["s2"].astype(np.float32),
            s1_real=s["s1"].astype(np.float32),
            s1_gen=gen_s1.astype(np.float32),
            dem=s["dem"].astype(np.float32),
            twi=twi.astype(np.float32),
        )
        done += 1
        print(f"[{done:>3}/{n_tiles}] {s['key']}  "
              f"gen_S1 range [{gen_s1.min():.2f}, {gen_s1.max():.2f}]")
        if done >= n_tiles:
            break

    print(f"Done: {done} tiles in {out_dir}/")


if __name__ == "__main__":
    main()
