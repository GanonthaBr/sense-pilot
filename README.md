# SENSE Pilot

Pilot study for **SENSE** (*Scientific Evaluation of Synthesized Earth-observation*) —
a physics-aware benchmark measuring whether generative EO models (TerraMind, RemoteBAGEL, ...)
produce **scientifically valid** outputs, not just visually plausible ones.

**The pilot question:** do generated tiles violate physical/statistical constraints
(terrain-consistent hydrology, spectral-index distributions, SAR speckle statistics)
that real imagery respects?

**The design:** for each metric, compare THREE score distributions —
`real-vs-real` (control floor) / `generated-vs-real` (test) / `shuffled` (broken ceiling).
Pre-registered decision rule: the gap is confirmed iff generated falls far outside
the real-vs-real range on ≥ 2 metric families.

## Repo layout

```
src/
  metrics.py          # TWI terrain-violation, Jensen-Shannon divergence, ENL  (seed of SENSE)
  inspect_sample.py   # stream 1 TerraMesh val sample; print band order / dB / DEM facts
  terramesh.py        # (downloaded, not committed) TerraMesh WebDataset loader
  smoke_test.py       # TerraMind loads + generates on a dummy tensor
  data.py             # TODO: tiles + DEM -> TWI
  generate.py         # TODO: batch TerraMind generation on N tiles
  evaluate.py         # TODO: 3-way control harness
configs/pilot.yaml    # paths, thresholds, decision rule
slurm/generate.sbatch # single-GPU batch job template (Bridges-2)
outputs/              # generated tiles, tables, figures (not committed)
```

## Setup on Bridges-2

```bash
module load anaconda3
conda activate $PROJECT/envs/terramind311     # Python 3.11 / torchgeo 0.8.1 / terratorch 1.2.8 / torch cu124
export HF_HOME=$PROJECT/.huggingface
export PIP_CACHE_DIR=$PROJECT/.pip-cache

pip install --no-cache-dir -r requirements.txt
wget https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh/resolve/main/terramesh.py -P src/
```

## Order of operations

1. `python src/smoke_test.py` — confirm TerraMind runs (CPU is fine).
2. `python src/inspect_sample.py` — on the LOGIN node (network); lock band order, dB, DEM.
3. `src/data.py` + `src/evaluate.py` — written against the inspected format.
4. `sbatch slurm/generate.sbatch` — 50-tile generation on a GPU node (data local first).
5. Metrics + contact sheet + the three-number table.

## Ground rules

- Never run GPU code on login nodes; never rely on network from compute nodes.
- Every run has a config; results without a config don't exist.
- Decision rules are written BEFORE looking at results.
- zarr==2.18.0 / numcodecs==0.15.1 are pinned per TerraMesh docs — do not upgrade.
