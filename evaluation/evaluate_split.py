import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data.atm_dataset import load_atm_dataset
from encoding.atm_encoder import encode_atm_sample
from models.atm_hybrid_model import ATMHybridSNNANN
from models.event_selector import EventSelector


TIME_STEPS = 20
TOP_K = 20
INPUT_SIZE = 128


# =========================================================
# Load data
# =========================================================

X, y = load_atm_dataset()


# Recreate EXACT same train/test split
indices = np.arange(len(X))

train_indices, test_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("=" * 70)
print("SPLIT-COMPUTING EVALUATION")
print("=" * 70)

print("Total samples :", len(X))
print("Train samples :", len(train_indices))
print("Test samples  :", len(test_indices))


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
# Event selector
# =========================================================

selector = EventSelector(
    top_k=TOP_K
)


# =========================================================
# Process one sample
# =========================================================

def process_sample(sample):

    # -----------------------------
    # Encode
    # -----------------------------

    spikes = encode_atm_sample(
        sample,
        time_steps=TIME_STEPS,
        seed=42
    )

    x = torch.tensor(
        spikes,
        dtype=torch.float32
    ).unsqueeze(1)

    # -----------------------------
    # SNN
    # -----------------------------

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

    # -----------------------------
    # Event selection
    # -----------------------------

    with torch.no_grad():

        selected_values, selected_indices, active = selector(
            snn_features
        )

    # -----------------------------
    # Reconstruct on server
    # -----------------------------

    reconstructed = torch.zeros(
        selected_values.size(0),
        INPUT_SIZE
    )

    reconstructed.scatter_(
        1,
        selected_indices,
        selected_values
    )

    # -----------------------------
    # ANN
    # -----------------------------

    with torch.no_grad():

        output = model.ann(
            reconstructed
        )

        prediction = output.argmax(
            dim=1
        ).item()

    # -----------------------------
    # Communication metrics
    # -----------------------------

    original_features = snn_features.numel()

    transmitted_features = (
        selected_values.numel()
    )

    reduction = (
        1 -
        transmitted_features /
        original_features
    ) * 100

    return (
        prediction,
        int(active.item()),
        original_features,
        transmitted_features,
        reduction
    )


# =========================================================
# Evaluate test set
# =========================================================

correct = 0

results = []

total_original = 0
total_transmitted = 0
total_reduction = 0


print()
print("-" * 70)
print("TEST RESULTS")
print("-" * 70)


for position, index in enumerate(test_indices):

    sample = X[index]

    true_label = y[index]

    (
        prediction,
        active,
        original_features,
        transmitted_features,
        reduction
    ) = process_sample(sample)

    correct_prediction = (
        prediction == true_label
    )

    if correct_prediction:
        correct += 1

    total_original += original_features
    total_transmitted += transmitted_features
    total_reduction += reduction

    results.append(
        (
            index,
            true_label,
            prediction,
            active,
            reduction,
            correct_prediction
        )
    )

    print(
        f"Sample {position + 1:02d} | "
        f"Index {index:02d} | "
        f"True: "
        f"{'LEFT ' if true_label == 0 else 'RIGHT'} | "
        f"Pred: "
        f"{'LEFT ' if prediction == 0 else 'RIGHT'} | "
        f"Active: {active:3d} | "
        f"Reduction: {reduction:6.2f}% | "
        f"{'CORRECT' if correct_prediction else 'WRONG'}"
    )


# =========================================================
# Final metrics
# =========================================================

test_accuracy = (
    100 * correct / len(test_indices)
)

average_reduction = (
    total_reduction /
    len(test_indices)
)

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    f"Correct predictions : "
    f"{correct}/{len(test_indices)}"
)

print(
    f"Test accuracy       : "
    f"{test_accuracy:.2f}%"
)

print(
    f"Average reduction   : "
    f"{average_reduction:.2f}%"
)

print(
    f"Original features   : "
    f"{total_original // len(test_indices)} per sample"
)

print(
    f"Transmitted values  : "
    f"{total_transmitted // len(test_indices)} per sample"
)

print("=" * 70)