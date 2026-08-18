import torch

from models.atm_hybrid_model import ATMHybridSNNANN
from data.atm_dataset import load_atm_dataset
from encoding.atm_encoder import encode_atm_sample


# Load ATM data
X, y = load_atm_dataset()

# Encode first sample
spikes = encode_atm_sample(
    X[0],
    time_steps=20
)

# Convert to tensor
input_tensor = torch.tensor(
    spikes,
    dtype=torch.float32
).unsqueeze(1)

print("=" * 60)
print("ATM HYBRID SNN + ANN TEST")
print("=" * 60)

print("Input shape:", input_tensor.shape)

# Create model
model = ATMHybridSNNANN()

# Forward pass
output, snn_features, spike_train = model(input_tensor)

print("SNN feature shape:", snn_features.shape)
print("ANN output shape :", output.shape)
print("Output:", output)

print("=" * 60)