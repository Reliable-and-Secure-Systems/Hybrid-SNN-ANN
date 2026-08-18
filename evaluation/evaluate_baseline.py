import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data.atm_dataset import load_atm_dataset
from encoding.atm_encoder import encode_atm_sample
from models.atm_hybrid_model import ATMHybridSNNANN


TIME_STEPS = 20


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
# Run SNN + ANN without compression
# =========================================================

def predict(sample):

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

        snn_features = spike_outputs.mean(
            dim=0
        )

        output = model.ann(
            snn_features
        )

        prediction = output.argmax(
            dim=1
        ).item()

    return prediction


# =========================================================
# Evaluate
# =========================================================

correct = 0

print("=" * 70)
print("BASELINE: SNN → ANN WITHOUT EVENT COMPRESSION")
print("=" * 70)

for position, index in enumerate(test_indices):

    prediction = predict(
        X[index]
    )

    true_label = y[index]

    is_correct = (
        prediction == true_label
    )

    if is_correct:
        correct += 1

    print(
        f"Sample {position + 1:02d} | "
        f"Index {index:02d} | "
        f"True: "
        f"{'LEFT ' if true_label == 0 else 'RIGHT'} | "
        f"Pred: "
        f"{'LEFT ' if prediction == 0 else 'RIGHT'} | "
        f"{'CORRECT' if is_correct else 'WRONG'}"
    )


accuracy = (
    100 * correct / len(test_indices)
)

print()
print("=" * 70)
print("BASELINE RESULTS")
print("=" * 70)

print(
    f"Correct predictions : "
    f"{correct}/{len(test_indices)}"
)

print(
    f"Test accuracy       : "
    f"{accuracy:.2f}%"
)

print(
    "Features transmitted: 128"
)

print("=" * 70)