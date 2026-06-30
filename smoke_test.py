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

dummy = torch.randn(1, 12, 224, 224)   # 12-band S2L2A
with torch.no_grad():
    out = model({'S2L2A': dummy})

print({k: v.shape for k, v in out.items()})
