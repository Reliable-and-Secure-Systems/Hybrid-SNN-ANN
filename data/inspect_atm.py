import matplotlib.pyplot as plt
import numpy as np

from data.atm_dataset import load_atm_dataset


# Load ATM data
X, y = load_atm_dataset()

# Take first trial
sample = X[0]

print("Sample shape:", sample.shape)
print("Label:", y[0])

# Plot the 68 x 68 representation
plt.figure(figsize=(8, 6))

plt.imshow(sample, aspect="auto")

plt.colorbar()

plt.title(
    f"ATM EEG Representation - "
    f"{'LEFT' if y[0] == 0 else 'RIGHT'}"
)

plt.xlabel("Brain Region")
plt.ylabel("Brain Region")

plt.tight_layout()

plt.show()