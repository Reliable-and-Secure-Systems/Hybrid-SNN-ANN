import torch

from data.atm_dataset import load_atm_dataset
from encoding.atm_encoder import encode_atm_sample
from models.atm_hybrid_model import ATMHybridSNNANN
from models.event_selector import EventSelector


TIME_STEPS = 20
TOP_K = 20
INPUT_SIZE = 128


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
# Load ATM dataset
# =========================================================

X, y = load_atm_dataset()


# =========================================================
# Edge: SNN
# =========================================================

def edge_snn(sample):

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

    return snn_features


# =========================================================
# Event Selection
# =========================================================

selector = EventSelector(
    top_k=TOP_K
)


def select_events(features):

    with torch.no_grad():

        values, indices, active = selector(
            features
        )

    return values, indices, active


# =========================================================
# Server: reconstruct compressed data
# =========================================================

def reconstruct_features(
    values,
    indices
):

    reconstructed = torch.zeros(
        values.size(0),
        INPUT_SIZE
    )

    reconstructed.scatter_(
        1,
        indices,
        values
    )

    return reconstructed


# =========================================================
# Server: ANN
# =========================================================

def server_ann(features):

    with torch.no_grad():

        output = model.ann(
            features
        )

        prediction = output.argmax(
            dim=1
        )

    return prediction, output


# =========================================================
# Run one sample
# =========================================================

sample_index = 0

sample = X[sample_index]

true_label = y[sample_index]


print("=" * 70)
print("END-TO-END SPLIT COMPUTING")
print("=" * 70)


# -------------------------
# Edge
# -------------------------

snn_features = edge_snn(sample)

print("\nEDGE DEVICE")
print("Original ATM shape :", sample.shape)
print("SNN features       :", snn_features.shape)


# -------------------------
# Event selection
# -------------------------

selected_values, selected_indices, active = select_events(
    snn_features
)

print("\nEVENT SELECTION")
print("Active SNN features :", int(active.item()))
print("Selected features   :", selected_values.shape)


# -------------------------
# Transmission
# -------------------------

original_values = snn_features.numel()

transmitted_values = selected_values.numel()

reduction = (
    1 -
    transmitted_values / original_values
) * 100

print("\nTRANSMISSION")
print("Original values :", original_values)
print("Sent values     :", transmitted_values)
print(f"Value reduction : {reduction:.2f}%")


# -------------------------
# Server reconstruction
# -------------------------

reconstructed = reconstruct_features(
    selected_values,
    selected_indices
)

print("\nSERVER")
print("Reconstructed shape:", reconstructed.shape)


# -------------------------
# ANN prediction
# -------------------------

prediction, output = server_ann(
    reconstructed
)

predicted_label = prediction.item()


print("\nCLASSIFICATION")

print(
    "True label:",
    "LEFT" if true_label == 0 else "RIGHT"
)

print(
    "Predicted:",
    "LEFT" if predicted_label == 0 else "RIGHT"
)

print(
    "ANN output:",
    output
)

print("=" * 70)