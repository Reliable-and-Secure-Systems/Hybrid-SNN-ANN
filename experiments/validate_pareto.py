import random
import numpy as np
import torch
import pandas as pd

from sklearn.model_selection import train_test_split

from data.atm_dataset import load_atm_dataset
from encoding.atm_encoder import encode_atm_sample
from models.atm_hybrid_model import ATMHybridSNNANN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

TIME_STEPS = 10
BETA = 0.95

INPUT_SIZE = 68 * 68
HIDDEN_SIZE = 128
NUM_CLASSES = 2

SEEDS = [42, 123, 456, 789, 2026]

CONFIGS = [
    {
        "name": "Pareto_40bit",
        "split_point": "ann_hidden",
        "top_k": 10,
        "bits": 4,
    },
    {
        "name": "Pareto_192bit",
        "split_point": "ann_hidden",
        "top_k": 48,
        "bits": 4,
    },
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_dataset(X):
    encoded = []

    for sample in X:
        spikes = encode_atm_sample(
            sample,
            time_steps=TIME_STEPS,
            seed=42,
        )
        encoded.append(spikes)

    return torch.tensor(
        np.array(encoded),
        dtype=torch.float32,
    )


def train_model(train_x, train_y):
    model = ATMHybridSNNANN(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        beta=BETA,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(EPOCHS):
        permutation = torch.randperm(len(train_x))

        for start in range(0, len(train_x), BATCH_SIZE):
            indices = permutation[start:start + BATCH_SIZE]

            batch_x = train_x[indices].to(DEVICE)
            batch_y = train_y[indices].to(DEVICE)

            optimizer.zero_grad()

            output, _, _, _ = model(batch_x.transpose(0, 1))

            loss = criterion(output, batch_y)

            loss.backward()
            optimizer.step()

    return model


def apply_top_k(features, top_k):
    values = features.abs()

    _, indices = torch.topk(
        values,
        k=top_k,
        dim=1,
    )

    selected = torch.zeros_like(features)

    selected.scatter_(
        1,
        indices,
        features.gather(1, indices),
    )

    return selected


def classify_from_split(model, features, split_point):
    if split_point == "ann_hidden":
        output = model.ann_fc2(features)

    elif split_point == "snn_output":
        hidden = model.ann_relu(
            model.ann_fc1(features)
        )
        output = model.ann_fc2(hidden)

    else:
        raise ValueError(
            f"Unknown split point: {split_point}"
        )

    return output


def quantize_dequantize(values, bits):
    max_value = values.abs().max()

    if max_value == 0:
        return torch.zeros_like(values)

    max_int = (2 ** (bits - 1)) - 1

    scale = max_value / max_int

    quantized = torch.round(
        values / scale
    ).clamp(
        -max_int,
        max_int,
    )

    return quantized * scale


def evaluate_configuration(
    model,
    test_x,
    test_y,
    split_point,
    top_k,
    bits,
):
    model.eval()

    correct = 0
    total = len(test_x)

    with torch.no_grad():

        for start in range(0, total, BATCH_SIZE):

            batch_x = test_x[
                start:start + BATCH_SIZE
            ].to(DEVICE)

            batch_y = test_y[
                start:start + BATCH_SIZE
            ].to(DEVICE)

            _, snn_features, _, ann_hidden = model(
    batch_x.transpose(0, 1)
)

            if split_point == "snn_output":
                features = snn_features
                feature_count = HIDDEN_SIZE

            elif split_point == "ann_hidden":
                features = ann_hidden
                feature_count = 64

            else:
                raise ValueError(
                    f"Unknown split point: {split_point}"
                )

            features = apply_top_k(
                features,
                top_k,
            )

            features = quantize_dequantize(
                features,
                bits,
            )

            output = classify_from_split(
                model,
                features,
                split_point,
            )

            predictions = output.argmax(dim=1)

            correct += (
                predictions == batch_y
            ).sum().item()

    accuracy = correct / total

    transmitted_bits = top_k * bits

    return (
        accuracy,
        correct,
        transmitted_bits,
        feature_count,
    )


def main():

    print("=" * 70)
    print("PARETO MULTI-SEED VALIDATION")
    print("=" * 70)

    X, y = load_atm_dataset()

    print("Dataset:", X.shape)

    encoded = encode_dataset(X)

    print("Encoded:", encoded.shape)

    all_results = []

    for seed in SEEDS:

        print()
        print("=" * 70)
        print(f"SEED {seed}")
        print("=" * 70)

        set_seed(seed)

        indices = np.arange(len(y))

        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        train_x = encoded[train_idx]
        test_x = encoded[test_idx]

        train_y = torch.tensor(
            y[train_idx],
            dtype=torch.long,
        )

        test_y = torch.tensor(
            y[test_idx],
            dtype=torch.long,
        )

        model = train_model(
            train_x,
            train_y,
        )

        for config in CONFIGS:

            accuracy, correct, transmitted_bits, feature_count = (
                evaluate_configuration(
                    model,
                    test_x,
                    test_y,
                    config["split_point"],
                    config["top_k"],
                    config["bits"],
                )
            )

            print(
                f'{config["name"]}: '
                f'{accuracy * 100:.2f}% '
                f'({correct}/{len(test_y)}) | '
                f'bits={transmitted_bits}'
            )

            all_results.append({
                "seed": seed,
                "configuration": config["name"],
                "time_steps": TIME_STEPS,
                "beta": BETA,
                "split_point": config["split_point"],
                "top_k": config["top_k"],
                "quantization_bits": config["bits"],
                "accuracy": accuracy * 100,
                "correct": correct,
                "total": len(test_y),
                "transmitted_bits": transmitted_bits,
            })

    results = pd.DataFrame(all_results)

    results.to_csv(
        "results/pareto_validation.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = (
        results
        .groupby("configuration")
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            min_accuracy=("accuracy", "min"),
            max_accuracy=("accuracy", "max"),
            transmitted_bits=("transmitted_bits", "first"),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    summary.to_csv(
        "results/pareto_validation_summary.csv",
        index=False,
    )

    print()
    print("Saved:")
    print("  results/pareto_validation.csv")
    print("  results/pareto_validation_summary.csv")


if __name__ == "__main__":
    main()