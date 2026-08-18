import torch

from snn_model import SNNModel

model = SNNModel()

dummy_input = torch.rand(1, 500)

output = model(dummy_input)

print("=" * 40)
print("Input Shape :", dummy_input.shape)
print("Output Shape:", output.shape)
print(output)
print("=" * 40)