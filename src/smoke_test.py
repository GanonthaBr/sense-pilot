"""
smoke_test.py — Minimal end-to-end check that TerraMind loads and generates.
Passed on Bridges-2 (2026-07): output {'S1GRD': torch.Size([1, 2, 224, 224])}.

Runs on CPU deliberately (device-placement of TerraMind's normalization buffers
is handled later in generate.py; a smoke test only needs the forward pass to work).
"""
import torch
from terratorch import FULL_MODEL_REGISTRY

print("CUDA available:", torch.cuda.is_available())

model = FULL_MODEL_REGISTRY.build(
    'terramind_v1_base_generate',
    pretrained=True,
    modalities=['S2L2A'],
    output_modalities=['S1GRD'],
    timesteps=10,
    standardize=True,
)
model = model.eval()

dummy = torch.randn(1, 12, 224, 224)
with torch.no_grad():
    out = model({'S2L2A': dummy})

print({k: v.shape for k, v in out.items()})
