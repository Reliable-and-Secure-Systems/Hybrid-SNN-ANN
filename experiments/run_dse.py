import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from data.atm_dataset import load_atm_dataset
from encoding.atm_encoder import encode_atm_sample
from models.atm_hybrid_model import ATMHybridSNNANN
from split_computing.quantization import (
    quantize_tensor,
    dequantize_tensor,
)


# ============================================================
# Configuration
# ============================================================

import argparse

parser = argparse.ArgumentParser(
    description="Run SNN-ANN design space exploration"
)

parser.add_argument("--time-steps", type=int, default=10)
parser.add_argument("--beta", type=float, default=0.8)
parser.add_argument("--epochs", type=int, default=20)

args = parser.parse_args()

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 16
EPOCHS = args.epochs
LEARNING_RATE = 0.001

TEST_SIZE = 0.2
RANDOM_STATE = 42

INPUT_SIZE = 68 * 68
HIDDEN_SIZE = 128
NUM_CLASSES = 2

TIME_STEPS = args.time_steps
BETA = args.beta

SPLIT_POINTS = [
    "snn_output",
    "ann_hidden",
]

TOP_K_VALUES = [
    128,
    96,
    64,
    48,
    32,
    20,
    10,
]

QUANTIZATION_BITS = [
    32,
    16,
    8,
    4,
]

MODEL_PATH = "models/dse_model.pt"
RESULT_PATH = f"results/dse_T{TIME_STEPS}_B{BETA}.csv"


# ============================================================
# Helper functions
# ============================================================

def select_top_k(features, k):
    """
    Select the k features with the largest absolute activation.
    """

    k = min(k, features.shape[1])

    _, indices = torch.topk(
        features.abs(),
        k=k,
        dim=1
    )

    selected = torch.gather(
        features,
        1,
        indices
    )

    return selected


def quantize_for_transmission(features, bits):
    """
    Quantise features for communication.

    32 bits means FP32 and therefore no quantisation.
    """

    if bits == 32:
        return features, 32

    quantized, scale = quantize_tensor(
        features,
        bits
    )

    reconstructed = dequantize_tensor(
        quantized,
        scale
    )

    return reconstructed, bits


def get_split_features(
    model,
    batch_x,
    split_point
):
    """
    Run the model and return the representation
    that would be transmitted at the selected split point.
    """

    output, snn_features, spikes, ann_hidden = model(
        batch_x
    )

    if split_point == "snn_output":
        features = snn_features

    elif split_point == "ann_hidden":
        features = ann_hidden

    else:
        raise ValueError(
            f"Unknown split point: {split_point}"
        )

    return output, features


# ============================================================
# Load dataset
# ============================================================

print("=" * 70)
print("DESIGN SPACE EXPLORATION")
print("=" * 70)

print("Device:", DEVICE)

X, y = load_atm_dataset()

print()
print("Dataset shape:", X.shape)
print("Labels shape :", y.shape)

train_x_raw, test_x_raw, train_y, test_y = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print()
print("Training samples:", len(train_y))
print("Testing samples :", len(test_y))


# ============================================================
# Encode dataset
# ============================================================

print()
print("=" * 70)
print("ENCODING")
print("=" * 70)

train_encoded = []

for i, sample in enumerate(train_x_raw):

    spikes = encode_atm_sample(
        sample,
        time_steps=TIME_STEPS,
        seed=42 + i
    )

    train_encoded.append(spikes)


test_encoded = []

for i, sample in enumerate(test_x_raw):

    spikes = encode_atm_sample(
        sample,
        time_steps=TIME_STEPS,
        seed=10000 + i
    )

    test_encoded.append(spikes)


train_encoded = torch.tensor(
    np.array(train_encoded),
    dtype=torch.float32
)

test_encoded = torch.tensor(
    np.array(test_encoded),
    dtype=torch.float32
)

train_labels = torch.tensor(
    train_y,
    dtype=torch.long
)

test_labels = torch.tensor(
    test_y,
    dtype=torch.long
)

print(
    "Training encoded shape:",
    train_encoded.shape
)

print(
    "Testing encoded shape :",
    test_encoded.shape
)


# ============================================================
# Train one model
# ============================================================

print()
print("=" * 70)
print("TRAINING BASE MODEL")
print("=" * 70)

model = ATMHybridSNNANN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_classes=NUM_CLASSES,
    beta=BETA
).to(DEVICE)


train_dataset = TensorDataset(
    train_encoded,
    train_labels
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        # time × batch × features
        batch_x = batch_x.permute(
            1,
            0,
            2
        )

        optimizer.zero_grad()

        output, _, _, _ = model(
            batch_x
        )

        loss = criterion(
            output,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = output.argmax(
            dim=1
        )

        correct += (
            predictions == batch_y
        ).sum().item()

        total += batch_y.size(0)

    train_accuracy = (
        100 * correct / total
    )

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )


# Save model

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print()
print("Model saved to:", MODEL_PATH)


# ============================================================
# Baseline test accuracy
# ============================================================

print()
print("=" * 70)
print("BASELINE EVALUATION")
print("=" * 70)

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for batch_start in range(
        0,
        len(test_encoded),
        BATCH_SIZE
    ):

        batch_x = test_encoded[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        batch_y = test_labels[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        batch_x = batch_x.permute(
            1,
            0,
            2
        )

        output, _, _, _ = model(
            batch_x
        )

        predictions = output.argmax(
            dim=1
        )

        correct += (
            predictions == batch_y
        ).sum().item()

        total += batch_y.size(0)


baseline_accuracy = (
    100 * correct / total
)

print(
    f"Baseline accuracy: "
    f"{baseline_accuracy:.2f}% "
    f"({correct}/{total})"
)


# ============================================================
# Design-space evaluation
# ============================================================

print()
print("=" * 70)
print("EVALUATING DESIGN SPACE")
print("=" * 70)


def apply_top_k(features, k):
    """
    Keep the k strongest features while preserving
    the original feature dimensionality.

    Non-selected features are set to zero.
    """

    k = min(k, features.shape[1])

    _, indices = torch.topk(
        features.abs(),
        k=k,
        dim=1
    )

    compressed = torch.zeros_like(features)

    compressed.scatter_(
        1,
        indices,
        torch.gather(
            features,
            1,
            indices
        )
    )

    return compressed


def apply_quantization(features, bits):
    """
    Quantise and dequantise the transmitted representation.

    FP32 is represented by 32 bits and is left unchanged.
    """

    if bits == 32:
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


def classify_from_split(
    model,
    features,
    split_point
):
    """
    Continue the ANN computation after the selected
    split point.
    """

    if split_point == "snn_output":

        ann_hidden = model.ann_relu(
            model.ann_fc1(features)
        )

        output = model.ann_fc2(
            ann_hidden
        )

    elif split_point == "ann_hidden":

        output = model.ann_fc2(
            features
        )

    else:

        raise ValueError(
            f"Unknown split point: {split_point}"
        )

    return output


results = []


with torch.no_grad():

    for split_point in SPLIT_POINTS:

        print()
        print(
            "Split point:",
            split_point
        )

        # ----------------------------------------------------
        # Generate features at the selected split point
        # ----------------------------------------------------

        all_features = []

        for batch_start in range(
            0,
            len(test_encoded),
            BATCH_SIZE
        ):

            batch_x = test_encoded[
                batch_start:
                batch_start + BATCH_SIZE
            ]

            batch_x = batch_x.to(DEVICE)

            # time × batch × features
            batch_x = batch_x.permute(
                1,
                0,
                2
            )

            _, snn_features, _, ann_hidden = model(
                batch_x
            )

            if split_point == "snn_output":

                features = snn_features

            elif split_point == "ann_hidden":

                features = ann_hidden

            else:

                raise ValueError(
                    f"Unknown split point: {split_point}"
                )

            all_features.append(
                features.cpu()
            )

        all_features = torch.cat(
            all_features,
            dim=0
        )

        feature_count = all_features.shape[1]

        print(
            "Feature count:",
            feature_count
        )

        # ----------------------------------------------------
        # Compression + quantisation sweep
        # ----------------------------------------------------

        for top_k in TOP_K_VALUES:

            if top_k > feature_count:
                continue

            compressed = apply_top_k(
                all_features,
                top_k
            )

            for bits in QUANTIZATION_BITS:

                transmitted = apply_quantization(
                    compressed,
                    bits
                )

                # ------------------------------------------------
                # Server-side ANN inference
                # ------------------------------------------------

                server_outputs = []

                for batch_start in range(
                    0,
                    len(transmitted),
                    BATCH_SIZE
                ):

                    batch_features = transmitted[
                        batch_start:
                        batch_start + BATCH_SIZE
                    ].to(DEVICE)

                    output = classify_from_split(
                        model,
                        batch_features,
                        split_point
                    )

                    server_outputs.append(
                        output.cpu()
                    )

                outputs = torch.cat(
                    server_outputs,
                    dim=0
                )

                predictions = outputs.argmax(
                    dim=1
                )

                correct_config = (
                    predictions == test_labels
                ).sum().item()

                total_config = len(
                    test_labels
                )

                accuracy = (
                    100 *
                    correct_config /
                    total_config
                )

                # ------------------------------------------------
                # Communication cost
                # ------------------------------------------------

                transmitted_bits = (
                    top_k * bits
                )

                original_bits = (
                    feature_count * 32
                )

                feature_reduction = (
                    100 *
                    (1 - top_k / feature_count)
                )

                bit_reduction = (
                    100 *
                    (1 - transmitted_bits / original_bits)
                )

                results.append({

                    "time_steps":
                        TIME_STEPS,

                    "beta":
                        BETA,

                    "split_point":
                        split_point,

                    "top_k":
                        top_k,

                    "quantization_bits":
                        bits,

                    "feature_count":
                        feature_count,

                    "accuracy":
                        accuracy,

                    "correct":
                        correct_config,

                    "total":
                        total_config,

                    "original_bits":
                        original_bits,

                    "transmitted_bits":
                        transmitted_bits,

                    "feature_reduction_percent":
                        feature_reduction,

                    "bit_reduction_percent":
                        bit_reduction
                })

                print(
                    f"  top_k={top_k:3d} | "
                    f"bits={bits:2d} | "
                    f"transmitted={transmitted_bits:5d} | "
                    f"accuracy={accuracy:.2f}% "
                    f"({correct_config}/{total_config})"
                )


# ============================================================
# Save results
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULT_PATH,
    index=False
)

print()
print("=" * 70)
print("DSE COMPLETE")
print("=" * 70)

print(
    "Configurations:",
    len(results_df)
)

print(
    "Results saved to:",
    RESULT_PATH
)

print()

print(
    results_df.to_string(
        index=False
    )
)