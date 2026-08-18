import torch

from ann_model import ANNModel

model = ANNModel()

dummy_features = torch.rand(1,64)

output = model(dummy_features)

print("="*40)
print("Input Shape :", dummy_features.shape)
print("Output Shape:", output.shape)
print(output)
print("="*40)