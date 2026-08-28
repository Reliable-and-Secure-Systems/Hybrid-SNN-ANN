import torch

from data.atm_dataset import load_atm_dataset
from encoding.atm_encoder import encode_atm_sample
from models.atm_hybrid_model import ATMHybridSNNANN
from models.event_selector import EventSelector
from split_computing.quantization import (
    quantize_tensor,
    dequantize_tensor
)


TIME_STEPS = 20
TOP_K = 20
QUANTIZATION_BITS = 8
INPUT_SIZE = 128


model = ATMHybridSNNANN()

model.load_state_dict(
    torch.load(
        "models/atm_hybrid_trained.pt",
        map_location="cpu"
    )
)

model.eval()


X, y = load_atm_dataset()



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


selector = EventSelector(
    top_k=TOP_K
)


def select_events(features):

    with torch.no_grad():

        values, indices, active = selector(
            features
        )

    return values, indices, active


def quantize_events(values):

    quantized, scale = quantize_tensor(
        values,
        QUANTIZATION_BITS
    )

    return quantized, scale



def dequantize_events(
    quantized_values,
    scale
):

    return dequantize_tensor(
        quantized_values,
        scale
    )


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


def server_ann(features):

    with torch.no_grad():

        output = model.ann(
            features
        )

        prediction = output.argmax(
            dim=1
        )

    return prediction, output


sample_index = 0

sample = X[sample_index]

true_label = y[sample_index]


print("=" * 75)
print("END-TO-END SPLIT COMPUTING")
print("=" * 75)



snn_features = edge_snn(sample)

print("\nEDGE DEVICE")
print("Original ATM shape :", sample.shape)
print("SNN features       :", snn_features.shape)


selected_values, selected_indices, active = select_events(
    snn_features
)

print("\nEVENT SELECTION")
print("Active SNN features :", int(active.item()))
print("Selected features   :", selected_values.shape)

quantized_values, scale = quantize_events(
    selected_values
)

print("\nQUANTISATION")
print("Precision           :", f"INT{QUANTIZATION_BITS}")
print("Scale               :", scale.item())
print("Quantized dtype     :", quantized_values.dtype)


original_bits = (
    snn_features.numel() * 32
)

transmitted_bits = (
    quantized_values.numel()
    * QUANTIZATION_BITS
)

bit_reduction = (
    1 -
    transmitted_bits / original_bits
) * 100

feature_reduction = (
    1 -
    selected_values.numel() /
    snn_features.numel()
) * 100

print("\nTRANSMISSION")
print("Original values     :", snn_features.numel())
print("Selected values     :", selected_values.numel())
print("Feature reduction   :", f"{feature_reduction:.2f}%")
print("Original bits       :", original_bits)
print("Transmitted bits    :", transmitted_bits)
print("Bit reduction      :", f"{bit_reduction:.2f}%")


dequantized_values = dequantize_events(
    quantized_values,
    scale
)

print("\nSERVER")
print("Dequantized values  :", dequantized_values.shape)


reconstructed = reconstruct_features(
    dequantized_values,
    selected_indices
)

print("Reconstructed shape :", reconstructed.shape)


prediction, output = server_ann(
    reconstructed
)

predicted_label = prediction.item()


print("\nCLASSIFICATION")

print(
    "True label          :",
    "LEFT" if true_label == 0 else "RIGHT"
)

print(
    "Predicted label     :",
    "LEFT" if predicted_label == 0 else "RIGHT"
)

print(
    "ANN output          :",
    output
)

print("=" * 75)