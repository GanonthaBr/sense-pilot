"""
metrics.py — Scientific-fidelity metrics for the SENSE pilot.

These functions measure whether a *generated* Earth-observation tile respects
physical/statistical properties that *real* imagery obeys. They are written to be
reused as the seed of the SENSE benchmark, so they are deliberately:
  - pure functions (no global state),
  - framework-light (numpy in, scalar/array out),
  - documented with units and assumptions.

Three metrics, one per physical claim:
  1. terrain_violation_score  -> hydrology: is generated water on impossible terrain?
  2. spectral_js_divergence   -> agriculture: do generated NDVI/NDWI distributions match real?
  3. enl                       -> SAR: does generated radar have correct speckle statistics?

A higher terrain_violation_score is WORSE. A higher spectral_js_divergence is WORSE.
ENL is compared against the real-vs-real range (closeness to real ENL is what matters).

Author: Bruno Ganontha — SENSE pilot.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Spectral indices (helpers) — computed from multi-spectral bands.
# ---------------------------------------------------------------------------
# NDVI = (NIR - RED) / (NIR + RED)        vegetation
# NDWI = (GREEN - NIR) / (GREEN + NIR)    open water (McFeeters 1996)
#
# IMPORTANT: band indices depend on the data source. For Sentinel-2 L2A the
# common ordering used by SSL4EO-S12 / TerraMind is 12 bands:
#   [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12]
#    0    1    2    3    4    5    6    7   8    9   10   11
# so GREEN=B03=index 2, RED=B04=index 3, NIR=B08=index 7.
# These defaults are set below but can be overridden — VERIFY against your data.
# ---------------------------------------------------------------------------

S2_GREEN_IDX = 2   # B03
S2_RED_IDX = 3     # B04
S2_NIR_IDX = 7     # B08
_EPS = 1e-6        # avoids divide-by-zero in index ratios


def compute_ndvi(tile: np.ndarray, nir_idx: int = S2_NIR_IDX, red_idx: int = S2_RED_IDX) -> np.ndarray:
    """NDVI map from a multi-band tile of shape (C, H, W). Returns (H, W) in [-1, 1]."""
    nir = tile[nir_idx].astype(np.float64)
    red = tile[red_idx].astype(np.float64)
    return (nir - red) / (nir + red + _EPS)


def compute_ndwi(tile: np.ndarray, green_idx: int = S2_GREEN_IDX, nir_idx: int = S2_NIR_IDX) -> np.ndarray:
    """NDWI map from a multi-band tile of shape (C, H, W). Returns (H, W) in [-1, 1].
    Positive NDWI generally indicates open water."""
    green = tile[green_idx].astype(np.float64)
    nir = tile[nir_idx].astype(np.float64)
    return (green - nir) / (green + nir + _EPS)


# ---------------------------------------------------------------------------
# METRIC 1 — Terrain / hydrology violation score
# ---------------------------------------------------------------------------
def terrain_violation_score(
    water_mask: np.ndarray,
    twi: np.ndarray,
    twi_threshold: float,
) -> float:
    """
    Fraction of predicted WATER pixels that sit on topographically implausible
    terrain (low Topographic Wetness Index -> ridgelines / steep dry slopes).

    Physical logic: real open water accumulates in low-lying, high-TWI areas.
    Water generated on low-TWI terrain (ridges) is a physical impossibility a
    purely data-driven model can produce. Real imagery scores near 0.

    Parameters
    ----------
    water_mask : (H, W) bool/0-1 array — pixels the model says are water.
    twi        : (H, W) float array  — Topographic Wetness Index, co-registered
                 to the SAME extent/resolution as water_mask.
    twi_threshold : float — pixels with TWI < threshold are "dry/elevated",
                 so water there is a violation. SET THIS FROM DATA (see note below),
                 do not hard-code arbitrarily.

    Returns
    -------
    float in [0, 1] — fraction of water pixels in violation. Higher = worse.
                      Returns 0.0 if there are no water pixels (nothing to violate).
    """
    water_mask = np.asarray(water_mask).astype(bool)
    twi = np.asarray(twi, dtype=np.float64)
    if water_mask.shape != twi.shape:
        raise ValueError(f"shape mismatch: water_mask {water_mask.shape} vs twi {twi.shape}")

    n_water = int(water_mask.sum())
    if n_water == 0:
        return 0.0

    violating = water_mask & (twi < twi_threshold)
    return float(violating.sum()) / float(n_water)


def water_mask_from_ndwi(ndwi: np.ndarray, ndwi_threshold: float = 0.0) -> np.ndarray:
    """Threshold an NDWI map into a binary water mask. NDWI > threshold => water.
    0.0 is the classic McFeeters cutoff; tune per scene if needed."""
    return (np.asarray(ndwi, dtype=np.float64) > ndwi_threshold)


def calibrate_twi_threshold(twi: np.ndarray, real_water_mask: np.ndarray, percentile: float = 5.0) -> float:
    """
    Derive twi_threshold from REAL data instead of guessing.

    Idea: look at the TWI values where real water actually occurs, and take a low
    percentile of that distribution as the floor. Any generated water below this
    floor is implausible. This makes the metric defensible ("threshold calibrated
    to the 5th percentile of real water's TWI") rather than arbitrary.

    Parameters
    ----------
    twi : (H, W) TWI map.
    real_water_mask : (H, W) bool — ground-truth water in the same scene.
    percentile : low percentile of real-water TWI to use as the threshold.

    Returns
    -------
    float threshold. Falls back to the global TWI median if no real water present.
    """
    twi = np.asarray(twi, dtype=np.float64)
    real_water_mask = np.asarray(real_water_mask).astype(bool)
    real_twi_values = twi[real_water_mask]
    if real_twi_values.size == 0:
        return float(np.median(twi))
    return float(np.percentile(real_twi_values, percentile))


# ---------------------------------------------------------------------------
# METRIC 2 — Spectral Jensen-Shannon divergence
# ---------------------------------------------------------------------------
def spectral_js_divergence(
    index_gen: np.ndarray,
    index_real: np.ndarray,
    n_bins: int = 64,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> float:
    """
    Jensen-Shannon divergence between the distribution of a spectral index
    (e.g. NDVI or NDWI) in the GENERATED tile vs the REAL tile.

    JS is chosen over KL because it is symmetric and bounded in [0, 1] (log base 2),
    so scores are comparable across tiles and never blow up to infinity.

    Parameters
    ----------
    index_gen  : array of generated index values (any shape; flattened internally).
    index_real : array of real index values.
    n_bins     : histogram resolution.
    value_range: range to bin over; NDVI/NDWI live in [-1, 1].

    Returns
    -------
    float in [0, 1]. 0 = identical distributions. Higher = worse (more divergence).
    """
    gen = np.asarray(index_gen, dtype=np.float64).ravel()
    real = np.asarray(index_real, dtype=np.float64).ravel()
    gen = gen[np.isfinite(gen)]
    real = real[np.isfinite(real)]
    if gen.size == 0 or real.size == 0:
        return float("nan")

    bins = np.linspace(value_range[0], value_range[1], n_bins + 1)
    p, _ = np.histogram(gen, bins=bins, density=False)
    q, _ = np.histogram(real, bins=bins, density=False)

    # normalize to probability distributions (+ tiny epsilon to avoid log(0))
    p = p.astype(np.float64) + _EPS
    q = q.astype(np.float64) + _EPS
    p /= p.sum()
    q /= q.sum()

    m = 0.5 * (p + q)
    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(js)  # base-2 KL below => JS in [0, 1]


def _kl(a: np.ndarray, b: np.ndarray) -> float:
    """KL(a || b) in bits (log base 2). Inputs must be normalized, strictly positive."""
    return float(np.sum(a * np.log2(a / b)))


# ---------------------------------------------------------------------------
# METRIC 3 — Equivalent Number of Looks (SAR speckle statistic)
# ---------------------------------------------------------------------------
def enl(sar_intensity: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    Equivalent Number of Looks over a (presumed homogeneous) region of a SAR
    intensity image. ENL = mean^2 / variance.

    Physical logic: real SAR speckle follows a known multiplicative-noise model,
    giving a characteristic ENL for a given sensor/processing. A generator that
    renders SAR as smooth pseudo-optical imagery will have a WRONG (usually much
    higher) ENL because it lacks real speckle texture. The pilot compares the ENL
    of generated SAR against the ENL of real SAR over matched regions.

    Parameters
    ----------
    sar_intensity : (H, W) SAR intensity (linear power, NOT dB). If your data is in
                    dB, convert first: intensity = 10 ** (db / 10).
    mask          : optional (H, W) bool — restrict to a homogeneous patch.

    Returns
    -------
    float ENL. Compared to real-vs-real ENL range; closeness to real is what matters.
    """
    x = np.asarray(sar_intensity, dtype=np.float64)
    if mask is not None:
        x = x[np.asarray(mask).astype(bool)]
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    mean = x.mean()
    var = x.var()
    if var <= 0:
        return float("inf")  # perfectly flat => no speckle => suspicious
    return float(mean * mean / var)


# ---------------------------------------------------------------------------
# quick self-test (run `python metrics.py` to sanity-check the math)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # JS: identical distributions -> ~0; different -> larger
    a = rng.normal(0.2, 0.1, 10000)
    b = rng.normal(0.2, 0.1, 10000)
    c = rng.normal(-0.5, 0.3, 10000)
    print("JS(real,real) ~0 :", round(spectral_js_divergence(a, b), 4))
    print("JS(real,fake) big:", round(spectral_js_divergence(a, c), 4))

    # terrain: water all on low-TWI -> ~1.0; water on high-TWI -> ~0.0
    twi = np.zeros((10, 10)); twi[:5] = 10.0      # top half wet, bottom half dry
    wet_water = np.zeros((10, 10), bool); wet_water[:5] = True
    dry_water = np.zeros((10, 10), bool); dry_water[5:] = True
    print("violation(valley water) ~0 :", terrain_violation_score(wet_water, twi, twi_threshold=5.0))
    print("violation(ridge water)  ~1 :", terrain_violation_score(dry_water, twi, twi_threshold=5.0))

    # ENL: speckled (low ENL) vs smoothed (high ENL)
    speckle = rng.gamma(shape=1.0, scale=1.0, size=(64, 64))   # heavy speckle, ENL~1
    smooth = np.full((64, 64), 1.0) + rng.normal(0, 0.01, (64, 64))  # nearly flat
    print("ENL(speckled) ~1   :", round(enl(speckle), 3))
    print("ENL(smoothed) huge :", round(enl(smooth), 1))
