import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data.atm_dataset import load_atm_dataset
from encoding.atm_encoder import encode_atm_sample
from models.atm_hybrid_model import ATMHybridSNNANN
from models.event_selector import EventSelector


TIME_STEPS = 20

TOP_K_VALUES = [
    128,
    96,
    64,
    48,
    32,
    20,
    10
]

INPUT_SIZE = 128


# =========================================================
# Load dataset
# =========================================================

X, y = load_atm_dataset()

indices = np.arange(len(X))

train_indices, test_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# Load trained model
# =========================================================

model = ATMHybridSNNANN()

model.load_state_dict(
    torch.load(
        "models/atm_hybrid_trained.pt",
        map_location="cpu"
    )
)

model.eval()


# =========================================================
# Generate SNN representation
# =========================================================

def get_snn_features(sample):

    spikes = encode_atm_sample(
        sample,
        time_steps=TIME_STEPS,
        seed=42
    )

    x = torch.tensor(
        spikes,
        dtype=torch.float32
    ).unsqueeze(1)

    with torch.no_grad():

        mem = model.snn_neuron.init_leaky()

        spike_outputs = []

        for t in range(x.size(0)):

            current = model.snn_fc(x[t])

            spike, mem = model.snn_neuron(
                current,
                mem
            )

            spike_outputs.append(spike)

        spike_outputs = torch.stack(
            spike_outputs
        )

        features = spike_outputs.mean(
            dim=0
        )

    return features


# =========================================================
# Evaluate each compression level
# =========================================================

print("=" * 75)
print("COMPRESSION vs ACCURACY EXPERIMENT")
print("=" * 75)

print(
    f"{'Features':>10} | "
    f"{'Reduction':>12} | "
    f"{'Correct':>8} | "
    f"{'Accuracy':>10}"
)

print("-" * 75)


for top_k in TOP_K_VALUES:

    selector = EventSelector(
        top_k=top_k
    )

    correct = 0

    for index in test_indices:

        sample = X[index]

        true_label = y[index]

        features = get_snn_features(
            sample
        )

        with torch.no_grad():

            values, indices_selected, _ = selector(
                features
            )

        # Reconstruct representation
        reconstructed = torch.zeros(
            1,
            INPUT_SIZE
        )

        reconstructed.scatter_(
            1,
            indices_selected,
            values
        )

        # ANN
        with torch.no_grad():

            output = model.ann(
                reconstructed
            )

            prediction = output.argmax(
                dim=1
            ).item()

        if prediction == true_label:
            correct += 1

    accuracy = (
        100 *
        correct /
        len(test_indices)
    )

    reduction = (
        1 -
        top_k / INPUT_SIZE
    ) * 100

    print(
        f"{top_k:>10} | "
        f"{reduction:>11.2f}% | "
        f"{correct:>5}/{len(test_indices)} | "
        f"{accuracy:>9.2f}%"
    )


print("=" * 75)