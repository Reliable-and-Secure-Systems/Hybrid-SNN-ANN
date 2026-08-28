import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data.atm_dataset import load_atm_dataset
from encoding.atm_encoder import encode_atm_sample
from models.atm_hybrid_model import ATMHybridSNNANN
from models.event_selector import EventSelector
from split_computing.quantization import (
    quantize_tensor,
    dequantize_tensor
)


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

PRECISION_VALUES = [
    None,
    16,
    8,
    4
]

INPUT_SIZE = 128
ORIGINAL_BITS = INPUT_SIZE * 32


X, y = load_atm_dataset()

indices = np.arange(len(X))

train_indices, test_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = ATMHybridSNNANN()

model.load_state_dict(
    torch.load(
        "models/atm_hybrid_trained.pt",
        map_location="cpu"
    )
)

model.eval()


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


def apply_quantization(features, bits):

    if bits is None:
        return features

    quantized, scale = quantize_tensor(
        features,
        bits
    )

    reconstructed = dequantize_tensor(
        quantized,
        scale
    )

    return reconstructed


print("=" * 90)
print("COMPRESSION vs QUANTISATION vs ACCURACY")
print("=" * 90)

print(
    f"{'Precision':>10} | "
    f"{'Features':>10} | "
    f"{'Feature Red.':>12} | {'Bit Red.':>10} | "
    f"{'Correct':>8} | "
    f"{'Accuracy':>10}"
)

print("-" * 90)


for bits in PRECISION_VALUES:

    precision_name = (
        "FP32" if bits is None
        else f"INT{bits}"
    )

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


            values = apply_quantization(
                values,
                bits
            )


            reconstructed = torch.zeros(
                1,
                INPUT_SIZE
            )

            reconstructed.scatter_(
                1,
                indices_selected,
                values
            )

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

        feature_reduction = (
            1 -
            top_k / INPUT_SIZE
        ) * 100

        bits_per_value = (
            32 if bits is None else bits
        )

        transmitted_bits = (
            top_k *
            bits_per_value
        )

        bit_reduction = (
            1 -
            transmitted_bits / ORIGINAL_BITS
        ) * 100

        print(
            f"{precision_name:>10} | "
            f"{top_k:>10} | "
            f"{feature_reduction:>11.2f}% | "
            f"{bit_reduction:>9.2f}% | "
            f"{correct:>5}/{len(test_indices)} | "
            f"{accuracy:>9.2f}%"
        )


print("=" * 90)