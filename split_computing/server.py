import torch

from models.ann_model import ANNModel

compressed = torch.load("compressed_events.pt")

print("Received Shape:", compressed.shape)

ann = ANNModel()

prediction = ann(compressed)

print("Prediction Shape:", prediction.shape)

print(prediction)