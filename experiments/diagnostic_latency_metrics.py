import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

from data.atm_dataset import load_atm_diagnostic_dataset
from encoding.atm_latency_encoder import encode_atm_latency
from models.atm_hybrid_model import ATMHybridSNNANN


TIME_STEPS = 20
BETA = 0.8
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

INPUT_SIZE = 68 * 68
HIDDEN_SIZE = 128
NUM_CLASSES = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 70)
print("LATENCY ENCODING - DETAILED DIAGNOSTIC EVALUATION")
print("=" * 70)

X, y, subjects, files = load_atm_diagnostic_dataset()

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

all_true = []
all_pred = []
fold_accuracies = []
fold_balanced_accuracies = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups=subjects),
    start=1
):

    print()
    print("=" * 70)
    print(f"FOLD {fold}/5")
    print("=" * 70)

    train_x = X[train_idx]
    test_x = X[test_idx]

    train_y = y[train_idx]
    test_y = y[test_idx]

    train_encoded = [
        encode_atm_latency(
            sample,
            time_steps=TIME_STEPS
        )
        for sample in train_x
    ]

    test_encoded = [
        encode_atm_latency(
            sample,
            time_steps=TIME_STEPS
        )
        for sample in test_x
    ]

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

    model = ATMHybridSNNANN(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        beta=BETA
    ).to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(
            train_encoded,
            train_labels
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        model.train()

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            batch_x = batch_x.permute(1, 0, 2)

            optimizer.zero_grad()

            output, _, _, _ = model(batch_x)

            loss = criterion(
                output,
                batch_y
            )

            loss.backward()
            optimizer.step()

    model.eval()

    predictions = []

    with torch.no_grad():

        for start in range(
            0,
            len(test_encoded),
            BATCH_SIZE
        ):

            batch_x = test_encoded[
                start:start + BATCH_SIZE
            ].to(DEVICE)

            batch_x = batch_x.permute(1, 0, 2)

            output, _, _, _ = model(batch_x)

            predictions.extend(
                output.argmax(dim=1)
                .cpu()
                .numpy()
            )

    predictions = np.array(predictions)

    accuracy = accuracy_score(
        test_y,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        test_y,
        predictions
    )

    fold_accuracies.append(accuracy)
    fold_balanced_accuracies.append(
        balanced_accuracy
    )

    all_true.extend(test_y)
    all_pred.extend(predictions)

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )


all_true = np.array(all_true)
all_pred = np.array(all_pred)


print()
print("=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"Mean Fold Accuracy: "
    f"{np.mean(fold_accuracies) * 100:.2f}%"
)

print(
    f"Fold Accuracy Std: "
    f"{np.std(fold_accuracies) * 100:.2f}%"
)

print(
    f"Mean Fold Balanced Accuracy: "
    f"{np.mean(fold_balanced_accuracies) * 100:.2f}%"
)

print(
    f"Balanced Accuracy Std: "
    f"{np.std(fold_balanced_accuracies) * 100:.2f}%"
)


print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        all_true,
        all_pred,
        target_names=["HC", "MCI", "AD"],
        digits=4
    )
)


print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    all_true,
    all_pred
)

print(cm)

print()
print("Rows = True class")
print("Columns = Predicted class")
print("Order = HC, MCI, AD")

print("=" * 70)
