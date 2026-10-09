import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, classification_report

from data.atm_dataset import load_atm_diagnostic_dataset


TIME_STEPS = 10
HIDDEN_SIZE = 128
NUM_CLASSES = 3
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.001


class DiagnosticANN(nn.Module):

    def __init__(
        self,
        input_size=4624,
        hidden_size=128,
        num_classes=3
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# =========================================================
# Load dataset
# =========================================================

X, y, subjects, files = load_atm_diagnostic_dataset()

print("=" * 70)
print("ANN-ONLY DIAGNOSTIC CROSS-VALIDATION")
print("=" * 70)

print(f"Samples : {len(X)}")
print(f"Subjects: {len(np.unique(subjects))}")

# Flatten 68x68 EEG matrices
X = X.reshape(len(X), -1)

print(f"Input shape: {X.shape}")


# =========================================================
# 5-fold subject-level cross-validation
# =========================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_accuracies = []


for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups=subjects),
    start=1
):

    print()
    print("=" * 70)
    print(f"FOLD {fold}")
    print("=" * 70)

    model = DiagnosticANN(
        input_size=4624,
        hidden_size=128,
        num_classes=3
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    X_train = torch.tensor(
        X[train_idx],
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y[train_idx],
        dtype=torch.long
    )

    X_test = torch.tensor(
        X[test_idx],
        dtype=torch.float32
    )

    y_test = torch.tensor(
        y[test_idx],
        dtype=torch.long
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    model.train()

    for epoch in range(EPOCHS):

        permutation = torch.randperm(
            len(X_train)
        )

        total_loss = 0.0
        correct = 0

        for start in range(
            0,
            len(X_train),
            BATCH_SIZE
        ):

            indices = permutation[
                start:start + BATCH_SIZE
            ]

            batch_x = X_train[indices]
            batch_y = y_train[indices]

            optimizer.zero_grad()

            output = model(batch_x)

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

        train_accuracy = (
            100 * correct / len(X_train)
        )

        if epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch + 1:02d} | "
                f"Loss: {total_loss:.4f} | "
                f"Train Accuracy: "
                f"{train_accuracy:.2f}%"
            )

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    model.eval()

    with torch.no_grad():

        output = model(X_test)

        predictions = output.argmax(
            dim=1
        ).numpy()

    accuracy = accuracy_score(
        y_test.numpy(),
        predictions
    ) * 100

    fold_accuracies.append(
        accuracy
    )

    print(
        f"Fold {fold} Accuracy: "
        f"{accuracy:.2f}%"
    )

    print(
        classification_report(
            y_test.numpy(),
            predictions,
            labels=[0, 1, 2],
            target_names=["HC", "MCI", "AD"],
            zero_division=0
        )
    )


# =========================================================
# Final results
# =========================================================

print()
print("=" * 70)
print("FINAL ANN-ONLY CROSS-VALIDATION RESULTS")
print("=" * 70)

for i, accuracy in enumerate(
    fold_accuracies,
    start=1
):
    print(
        f"Fold {i}: {accuracy:.2f}%"
    )

print()
print(
    f"Mean Accuracy: "
    f"{np.mean(fold_accuracies):.2f}%"
)

print(
    f"Std Accuracy: "
    f"{np.std(fold_accuracies):.2f}%"
)

print("=" * 70)