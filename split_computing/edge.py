import torch

from data.atm_dataset import load_atm_dataset
from encoding.atm_encoder import encode_atm_sample
from models.atm_hybrid_model import ATMHybridSNNANN
from models.event_selector import EventSelector


TIME_STEPS = 20
TOP_K = 20


def run_edge(sample):

    # --------------------------------------------------
    # Encode ATM sample
    # --------------------------------------------------

    spikes = encode_atm_sample(
        sample,
        time_steps=TIME_STEPS,
        seed=42
    )

    x = torch.tensor(
        spikes,
        dtype=torch.float32
    ).unsqueeze(1)

    # --------------------------------------------------
    # SNN
    # --------------------------------------------------

    model = ATMHybridSNNANN()

    model.load_state_dict(torch.load("models/atm_hybrid_trained.pt",map_location="cpu"))



    model.eval()

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

    # --------------------------------------------------
    # Event Selection
    # --------------------------------------------------

    selector = EventSelector(
        top_k=TOP_K
    )

    selected_features, selected_indices, active_features = selector(snn_features)

    return (
    snn_features,
    selected_features,
    selected_indices,
    active_features
    )


if __name__ == "__main__":

    X, y = load_atm_dataset()

    sample = X[0]

    (
    snn_features,
    selected_features,
    selected_indices,
    active_features
    ) = run_edge(sample)

    original_size = snn_features.numel()

    compressed_size = selected_features.numel()

    compression = (
        1 -
        compressed_size / original_size
    ) * 100

    print("=" * 60)
    print("EDGE DEVICE + EVENT SELECTION")
    print("=" * 60)

    print(
        "Original ATM sample:",
        sample.shape
    )

    print(
        "SNN features:",
        snn_features.shape
    )

    print(
        "Selected features:",
        selected_features.shape
    )

    print(
        "Values before transmission:",
        original_size
    )

    print(
        "Values after transmission:",
        compressed_size
    )

    print(
        f"Data reduction: {compression:.2f}%"
    )

    print(
        "Selected indices:",
        selected_indices
    )

    print(
    "Actually active SNN features:",
    int(active_features.item())
    )

    print("=" * 60)