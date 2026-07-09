"""
inspect_sample.py — Stream ONE TerraMesh val sample and print the interface facts
we need before writing evaluate.py:

  1. S2L2A: shape, dtype, per-band value stats  -> lock band count & ordering
  2. S1GRD: shape, dtype, value range           -> confirm dB (negative values)
  3. DEM:   shape, dtype, value range           -> confirm raw elevation (meters)

Run on the Bridges-2 LOGIN node (network is open there; no GPU needed):

    cd $PROJECT/sense-pilot
    python src/inspect_sample.py

Requires: pip install webdataset zarr==2.18.0 numcodecs==0.15.1
and src/terramesh.py downloaded from the TerraMesh HF repo:
    wget https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh/resolve/main/terramesh.py -P src/
"""

import sys
import numpy as np

try:
    from terramesh import build_terramesh_dataset
except ImportError as e:
    sys.exit(
        f"Import failed: {e}\n(If terramesh.py exists, the failure is a MISSING DEPENDENCY inside it — check the env is terramind311.)\n"
        "  wget https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh/resolve/main/terramesh.py -P src/\n"
        "and run this script from the folder that contains it (or add src/ to PYTHONPATH)."
    )

HF_PATH = "https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh/resolve/main/"
MODALITIES = ["S2L2A", "S1GRD", "DEM"]


def to_numpy(x):
    """Robustly convert tensor / xarray / numpy to a float numpy array."""
    if hasattr(x, "numpy"):          # torch tensor
        x = x.numpy()
    if hasattr(x, "values"):         # xarray
        x = x.values
    return np.asarray(x)


def describe(name: str, arr: np.ndarray, per_band: bool = False):
    print(f"\n=== {name} ===")
    print(f"  shape : {arr.shape}")
    print(f"  dtype : {arr.dtype}")
    a = arr.astype(np.float64)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        print("  !! no finite values")
        return
    print(f"  range : [{finite.min():.3f}, {finite.max():.3f}]")
    print(f"  mean  : {finite.mean():.3f}   std: {finite.std():.3f}")
    n_nan = int(np.size(a) - finite.size)
    if n_nan:
        print(f"  NaN/inf pixels: {n_nan}")

    if per_band:
        # assume band axis is the first axis of the last 3 dims: (C,H,W) or (T,C,H,W)
        band_axis_arr = a
        if band_axis_arr.ndim == 4:      # (T, C, H, W) -> take first timestep
            band_axis_arr = band_axis_arr[0]
            print("  (showing first timestep of a temporal stack)")
        if band_axis_arr.ndim == 3:
            print(f"  per-band stats ({band_axis_arr.shape[0]} bands):")
            for i in range(band_axis_arr.shape[0]):
                b = band_axis_arr[i]
                bf = b[np.isfinite(b)]
                print(f"    band {i:2d}: min={bf.min():9.3f}  max={bf.max():9.3f}  mean={bf.mean():9.3f}")


def main():
    print("Building streaming TerraMesh val dataset (this touches the network)...")
    dataset = build_terramesh_dataset(
        path=HF_PATH,
        modalities=MODALITIES,
        split="val",
        shuffle=False,       # deterministic: same first sample every run
        batch_size=1,        # loader requires an int
    )

    print("Pulling the first sample...")
    sample = next(iter(dataset))

    print(f"\nSample keys: {list(sample.keys())}")
    if "__key__" in sample:
        print(f"Sample id  : {sample['__key__']}")

    for mod in MODALITIES:
        if mod not in sample:
            print(f"\n!! modality '{mod}' missing from sample — keys are {list(sample.keys())}")
            continue
        arr = to_numpy(sample[mod])
        describe(mod, arr, per_band=(mod == "S2L2A"))

    # --- automated verdicts on the two facts we care about ---
    print("\n" + "=" * 50)
    print("VERDICTS")
    print("=" * 50)

    if "S1GRD" in sample:
        s1 = to_numpy(sample["S1GRD"]).astype(np.float64)
        s1f = s1[np.isfinite(s1)]
        if s1f.size and s1f.min() < 0:
            print("S1GRD contains negative values -> data is in dB.")
            print("  => convert before ENL: intensity = 10 ** (db / 10)")
        else:
            print("S1GRD is all non-negative -> likely LINEAR intensity; ENL can use it directly.")

    if "DEM" in sample:
        dem = to_numpy(sample["DEM"]).astype(np.float64)
        demf = dem[np.isfinite(dem)]
        if demf.size:
            lo, hi = demf.min(), demf.max()
            if -500 <= lo and hi <= 9000:
                print(f"DEM range [{lo:.1f}, {hi:.1f}] looks like raw elevation in meters -> derive TWI from it.")
            else:
                print(f"DEM range [{lo:.1f}, {hi:.1f}] is unusual -> check if normalized/scaled before computing TWI.")

    if "S2L2A" in sample:
        s2 = to_numpy(sample["S2L2A"])
        c = s2.shape[0] if s2.ndim == 3 else (s2.shape[1] if s2.ndim == 4 else None)
        print(f"S2L2A has {c} bands. If 12, expected order is")
        print("  [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12]")
        print("  => GREEN=idx2, RED=idx3, NIR=idx7 (the metrics.py defaults).")
        print("  Cross-check: the NIR band (idx 7) should have a HIGHER mean than RED (idx 3)")
        print("  over vegetated scenes — compare the per-band stats above.")


if __name__ == "__main__":
    main()
