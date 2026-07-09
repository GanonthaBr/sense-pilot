"""
data.py — TerraMesh data access + terrain preprocessing for the SENSE pilot.

Facts locked by src/inspect_sample.py on 2026-07-09 (Bridges-2):
  - Val split mixes 'majortom_val_*' (S2L2A+DEM) and 'ssl4eos12_val_*' (adds S1GRD).
    => we FILTER for samples containing S1GRD.
  - S2L2A: (1, 12, 264, 264) int16, reflectance x10000, band order
    [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12] -> GREEN=2, RED=3, NIR=7.
  - S1GRD: dB (loader standardization means are negative: VV -12.577, VH -20.265).
  - DEM: (1, 1, 264, 264) int16, raw elevation in meters.
  - Native 264x264 -> center-crop to 224 for TerraMind.
"""

from __future__ import annotations
import numpy as np

HF_PATH = "https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh/resolve/main/"
CROP = 224
NATIVE = 264


# ---------------------------------------------------------------------------
# sample handling
# ---------------------------------------------------------------------------
def to_numpy(x) -> np.ndarray:
    if hasattr(x, "numpy"):
        x = x.numpy()
    if hasattr(x, "values"):
        x = x.values
    return np.asarray(x)


def squeeze_batch(arr: np.ndarray) -> np.ndarray:
    """Drop the leading batch dim the loader adds with batch_size=1."""
    return arr[0] if arr.ndim >= 3 and arr.shape[0] == 1 else arr


def center_crop(arr: np.ndarray, size: int = CROP) -> np.ndarray:
    """Center-crop the last two axes to (size, size)."""
    h, w = arr.shape[-2], arr.shape[-1]
    top, left = (h - size) // 2, (w - size) // 2
    return arr[..., top:top + size, left:left + size]


def iter_paired_samples(path: str = HF_PATH, split: str = "val", max_samples: int | None = None):
    """
    Yield dicts {key, s2, s1, dem} for samples that contain ALL of S2L2A, S1GRD, DEM.
    Arrays are batch-squeezed, center-cropped to 224, and cast to float64.
    s2 stays in reflectance x10000 (ratios cancel in indices); s1 stays in dB.
    """
    from terramesh import build_terramesh_dataset  # local module, downloaded via wget

    ds = build_terramesh_dataset(
        path=path,
        modalities=["S2L2A", "S1GRD", "DEM"],
        split=split,
        shuffle=False,
        batch_size=1,
    )
    n = 0
    for sample in ds:
        if not all(m in sample for m in ("S2L2A", "S1GRD", "DEM")):
            continue  # majortom_* samples lack S1GRD — skip
        key = sample.get("__key__")
        key = key[0] if isinstance(key, (list, tuple)) else key
        s2 = center_crop(squeeze_batch(to_numpy(sample["S2L2A"]))).astype(np.float64)
        s1 = center_crop(squeeze_batch(to_numpy(sample["S1GRD"]))).astype(np.float64)
        dem = center_crop(squeeze_batch(to_numpy(sample["DEM"]))).astype(np.float64)
        dem = dem[0] if dem.ndim == 3 else dem  # (1,H,W) -> (H,W)
        yield {"key": key, "s2": s2, "s1": s1, "dem": dem}
        n += 1
        if max_samples is not None and n >= max_samples:
            return


# ---------------------------------------------------------------------------
# TWI from DEM — pure-numpy D8 implementation (pilot-grade).
# ---------------------------------------------------------------------------
# TWI = ln( a / tan(beta) )  with a = specific catchment area, beta = slope.
# D8 flow: each cell sends its accumulated area to its steepest-descent neighbor;
# processing cells in descending elevation order makes one pass sufficient.
# Pilot-grade note: no pit-filling; adequate for 224x224 patches. The full SENSE
# benchmark can swap in pysheds/richdem — the interface (dem -> twi map) is stable.
# ---------------------------------------------------------------------------
_D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def slope_radians(dem: np.ndarray, cellsize: float = 10.0) -> np.ndarray:
    """Slope (radians) from central differences. TerraMesh grids are 10 m."""
    gy, gx = np.gradient(dem.astype(np.float64), cellsize)
    return np.arctan(np.hypot(gx, gy))


def d8_flow_accumulation(dem: np.ndarray, cellsize: float = 10.0) -> np.ndarray:
    """Accumulated upslope area (m^2) via D8, one pass in descending elevation."""
    dem = dem.astype(np.float64)
    h, w = dem.shape
    acc = np.full((h, w), cellsize * cellsize)  # each cell contributes its own area
    order = np.argsort(dem, axis=None)[::-1]    # highest first
    dist = {d: cellsize * (np.hypot(*d)) for d in _D8}
    for flat in order:
        i, j = divmod(int(flat), w)
        zi = dem[i, j]
        best, best_drop = None, 0.0
        for di, dj in _D8:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w:
                drop = (zi - dem[ni, nj]) / dist[(di, dj)]
                if drop > best_drop:
                    best_drop, best = drop, (ni, nj)
        if best is not None:
            acc[best] += acc[i, j]
    return acc


def compute_twi(dem: np.ndarray, cellsize: float = 10.0) -> np.ndarray:
    """Topographic Wetness Index map. Higher = wetter/lower terrain."""
    beta = slope_radians(dem, cellsize)
    a = d8_flow_accumulation(dem, cellsize) / cellsize  # specific catchment area
    return np.log((a + 1e-6) / (np.tan(beta) + 1e-3))


if __name__ == "__main__":
    # tiny self-test on a synthetic valley: TWI must be higher at the valley floor
    y = np.abs(np.arange(64) - 32)
    dem = np.tile(y, (64, 1)).T.astype(float) * 5.0  # V-shaped valley along axis 0
    twi = compute_twi(dem)
    print("TWI valley floor (should be larger):", round(float(twi[32, 16:48].mean()), 2))
    print("TWI ridge        (should be smaller):", round(float(twi[2, 16:48].mean()), 2))
