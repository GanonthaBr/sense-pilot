"""
evaluate.py — The 3-way control harness for the SENSE pilot (S2 -> S1 task).

For each metric we report THREE score sets:
    real      : real S1 tiles (the physical floor)
    generated : TerraMind S1 tiles at the same locations (the thing tested)
    shuffled  : real S1 from tile i scored against context of tile j != i (broken ceiling)

Metrics (adapted to the S2->S1 generation task):
  1. ENL (VV) per tile          — speckle statistics; generated smooth SAR => huge ENL
  2. Backscatter JS (VV)        — distribution divergence vs the paired real tile
  3. SAR-water terrain violation— water (low VV) on topographically impossible terrain (TWI)
  4. Cross-modal water IoU      — does S1-derived water agree with real-S2 NDWI water?

Pre-registered decision rule (configs/pilot.yaml): the gap is confirmed iff the
generated scores fall clearly outside the real range on >= 2 metric families.

Usage:
    python src/evaluate.py [--tiles outputs/tiles] [--out outputs]
Produces:
    outputs/pilot_results.csv      (per-tile scores)
    outputs/pilot_summary.txt      (medians per condition + decision-rule verdict)
    outputs/contact_sheet.png      (real S2 RGB | real S1 VV | generated S1 VV)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from metrics import (
    enl, spectral_js_divergence, terrain_violation_score,
    compute_ndwi, water_mask_from_ndwi, calibrate_twi_threshold,
)

VV = 0  # S1GRD band order: [VV, VH]


# ---------------------------------------------------------------------------
# SAR helpers (task-specific; graduate into metrics.py for the full benchmark)
# ---------------------------------------------------------------------------
def db_to_linear(db: np.ndarray) -> np.ndarray:
    return 10.0 ** (np.asarray(db, dtype=np.float64) / 10.0)


def maybe_db_to_linear(arr: np.ndarray) -> np.ndarray:
    """Auto-detect units: negative values => dB => convert. Robust to a model
    that emits either dB-scaled or linear output."""
    a = np.asarray(arr, dtype=np.float64)
    return db_to_linear(a) if np.nanmin(a) < 0 else a


def water_mask_from_sar_vv(vv_db: np.ndarray, threshold_db: float = -18.0) -> np.ndarray:
    """Open water is smooth => very low backscatter. VV below ~-18 dB => water."""
    return np.asarray(vv_db, dtype=np.float64) < threshold_db


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")  # no water in either — uninformative, excluded from stats
    return float(np.logical_and(a, b).sum() / union)


# ---------------------------------------------------------------------------
def score_tile(s1_db: np.ndarray, ref_s1_db: np.ndarray, s2: np.ndarray,
               twi: np.ndarray, twi_thr: float) -> dict:
    """All four metrics for one S1 tile (real or generated) against its context."""
    vv = s1_db[VV]
    lin = maybe_db_to_linear(vv)
    ndwi_water = water_mask_from_ndwi(compute_ndwi(s2))
    sar_water = water_mask_from_sar_vv(vv)
    return {
        "enl_vv": enl(lin),
        "js_vv": spectral_js_divergence(vv, ref_s1_db[VV],
                                        value_range=(-35.0, 10.0)),
        "terrain_violation": terrain_violation_score(sar_water, twi, twi_thr),
        "water_iou": iou(sar_water, ndwi_water),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="outputs/tiles")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--factor", type=float, default=3.0,
                    help="decision rule: generated median beyond this factor of real median")
    args = ap.parse_args()

    tile_paths = sorted(Path(args.tiles).glob("*.npz"))
    if len(tile_paths) < 2:
        sys.exit(f"Need >=2 tiles in {args.tiles}; found {len(tile_paths)}. Run generate.py first.")
    print(f"Scoring {len(tile_paths)} tiles...")

    tiles = [dict(np.load(p)) | {"key": p.stem} for p in tile_paths]

    # calibrate the TWI threshold from REAL water across all tiles (defensible, not guessed)
    all_twi = np.concatenate([t["twi"].ravel() for t in tiles])
    all_real_water = np.concatenate(
        [water_mask_from_sar_vv(t["s1_real"][VV]).ravel() for t in tiles])
    twi_thr = calibrate_twi_threshold(all_twi, all_real_water, percentile=5.0)
    print(f"Calibrated TWI threshold (5th pct of real-water TWI): {twi_thr:.3f}")

    rows = []
    n = len(tiles)
    for i, t in enumerate(tiles):
        other = tiles[(i + 1) % n]  # deterministic mismatched partner for 'shuffled'
        common = dict(s2=t["s2"], twi=t["twi"], twi_thr=twi_thr)
        for cond, s1, ref in (
            ("real",      t["s1_real"], other["s1_real"]),   # real vs another real => natural JS floor
            ("generated", t["s1_gen"],  t["s1_real"]),        # generated vs ITS OWN real => the test
            ("shuffled",  other["s1_real"], t["s1_real"]),    # wrong-location real => broken ceiling
        ):
            rows.append({"key": t["key"], "condition": cond,
                         **score_tile(s1, ref, **common)})

    # ---- write CSV ----
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    metrics_list = ["enl_vv", "js_vv", "terrain_violation", "water_iou"]
    csv_path = out_dir / "pilot_results.csv"
    with open(csv_path, "w") as f:
        f.write("key,condition," + ",".join(metrics_list) + "\n")
        for r in rows:
            f.write(f"{r['key']},{r['condition']}," +
                    ",".join(f"{r[m]:.6f}" for m in metrics_list) + "\n")

    # ---- summary + decision rule ----
    def med(cond, m):
        v = np.array([r[m] for r in rows if r["condition"] == cond])
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else float("nan")

    lines = [f"SENSE pilot summary — {n} tiles, TWI thr {twi_thr:.3f}", ""]
    lines.append(f"{'metric':<20}{'real':>12}{'generated':>12}{'shuffled':>12}")
    exceed = 0
    for m in metrics_list:
        r_, g_, s_ = med("real", m), med("generated", m), med("shuffled", m)
        lines.append(f"{m:<20}{r_:>12.4f}{g_:>12.4f}{s_:>12.4f}")
        # direction-aware exceedance: iou LOWER is worse; others HIGHER is worse
        if m == "water_iou":
            bad = np.isfinite(r_) and np.isfinite(g_) and g_ < r_ / args.factor
        else:
            bad = np.isfinite(r_) and np.isfinite(g_) and g_ > r_ * args.factor
        if bad:
            exceed += 1
            lines.append(f"{'':<20}^ generated outside real range (factor {args.factor})")
    lines.append("")
    verdict = ("GAP CONFIRMED" if exceed >= 2 else "GAP NOT CONFIRMED") + \
              f" — {exceed} metric families exceed the pre-registered threshold (need >= 2)."
    lines.append(verdict)
    summary = "\n".join(lines)
    (out_dir / "pilot_summary.txt").write_text(summary)
    print("\n" + summary)

    # ---- contact sheet ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = min(6, n)
        fig, axes = plt.subplots(k, 3, figsize=(9, 3 * k))
        axes = np.atleast_2d(axes)
        for r_i in range(k):
            t = tiles[r_i]
            rgb = np.stack([t["s2"][3], t["s2"][2], t["s2"][1]], -1)  # B04,B03,B02
            rgb = np.clip(rgb / 3000.0, 0, 1)
            for c_i, (img, title, cmap) in enumerate((
                (rgb, "real S2 RGB", None),
                (t["s1_real"][VV], "real S1 VV (dB)", "gray"),
                (t["s1_gen"][VV], "generated S1 VV", "gray"),
            )):
                ax = axes[r_i, c_i]
                ax.imshow(img, cmap=cmap)
                ax.set_title(f"{t['key'][:22]}\n{title}", fontsize=7)
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "contact_sheet.png", dpi=150)
        print(f"Contact sheet: {out_dir/'contact_sheet.png'}")
    except Exception as e:  # noqa: BLE001 — contact sheet is best-effort
        print(f"(contact sheet skipped: {e})")


if __name__ == "__main__":
    main()
