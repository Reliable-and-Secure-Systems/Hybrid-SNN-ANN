import numpy as np

from data.atm_dataset import load_atm_dataset
from encoding.atm_encoder import encode_atm_sample


# Load real ATM data
X, y = load_atm_dataset()

# Select first real trial
sample = X[0]

# Encode it
spikes = encode_atm_sample(
    sample,
    time_steps=20
)

print("=" * 60)
print("REAL ATM SPIKE ENCODING")
print("=" * 60)

print("Original sample shape :", sample.shape)
print("Spike representation  :", spikes.shape)

print("Original non-zero values:",
      np.count_nonzero(sample))

print("Total generated spikes:",
      int(spikes.sum()))

print("Average spikes/timestep:",
      spikes.sum() / spikes.shape[0])

print("Label:",
      "LEFT" if y[0] == 0 else "RIGHT")

print("=" * 60)