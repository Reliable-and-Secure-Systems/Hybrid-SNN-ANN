import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score

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
print("DIAGNOSTIC EEG - LATENCY ENCODING EXPERIMENT")
print("=" * 70)

print("Device:", DEVICE)

X, y, subjects, files = load_atm_diagnostic_dataset()

print()
print("Dataset:", X.shape)
print("Subjects:", len(np.unique(subjects)))

print(
    "HC/MCI/AD:",
    np.sum(y == 0),
    np.sum(y == 1),
    np.sum(y == 2)
)


cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_accuracies = []
subject_results = []


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

    train_subjects = subjects[train_idx]
    test_subjects = subjects[test_idx]

    print(
        "Training subjects:",
        sorted(np.unique(train_subjects))
    )

    print(
        "Testing subjects :",
        sorted(np.unique(test_subjects))
    )

    print(
        "Testing HC/MCI/AD:",
        np.sum(test_y == 0),
        np.sum(test_y == 1),
        np.sum(test_y == 2)
    )

    print()
    print("Latency encoding...")

    train_encoded = []

    for sample in train_x:
        spikes = encode_atm_latency(
            sample,
            time_steps=TIME_STEPS
        )
        train_encoded.append(spikes)

    test_encoded = []

    for sample in test_x:
        spikes = encode_atm_latency(
            sample,
            time_steps=TIME_STEPS
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
        "Average train spikes/sample:",
        float(train_encoded.sum(dim=(1, 2)).mean())
    )

    print(
        "Average test spikes/sample:",
        float(test_encoded.sum(dim=(1, 2)).mean())
    )


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

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    print()
    print("Training...")

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

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

            total_loss += loss.item()

            predictions = output.argmax(dim=1)

            correct += (
                predictions == batch_y
            ).sum().item()

            total += batch_y.size(0)

        train_accuracy = (
            100 * correct / total
        )

        if epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch + 1:02d} | "
                f"Loss: "
                f"{total_loss / len(train_loader):.4f} | "
                f"Train Accuracy: "
                f"{train_accuracy:.2f}%"
            )


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

            batch_x = batch_x.permute(
                1,
                0,
                2
            )

            output, _, _, _ = model(
                batch_x
            )

            batch_predictions = (
                output.argmax(dim=1)
            )

            predictions.extend(
                batch_predictions.cpu().numpy()
            )


    predictions = np.array(predictions)

    accuracy = accuracy_score(
        test_y,
        predictions
    )

    fold_accuracies.append(accuracy)

    for subject in np.unique(test_subjects):

        subject_mask = (
            test_subjects == subject
        )

        subject_true = (
            test_y[subject_mask]
        )

        subject_pred = (
            predictions[subject_mask]
        )

        subject_accuracy = accuracy_score(
            subject_true,
            subject_pred
        )

        group_name = [
            "HC",
            "MCI",
            "AD"
        ][subject_true[0]]

        subject_results.append({
            "subject": subject,
            "group": group_name,
            "accuracy": subject_accuracy
        })

    print(
        f"Fold {fold} Accuracy: "
        f"{accuracy * 100:.2f}%"
    )


fold_accuracies = np.array(
    fold_accuracies
)

mean_accuracy = (
    fold_accuracies.mean() * 100
)

std_accuracy = (
    fold_accuracies.std() * 100
)


print()
print("=" * 70)
print("LATENCY ENCODING RESULTS")
print("=" * 70)

for i, accuracy in enumerate(
    fold_accuracies,
    start=1
):

    print(
        f"Fold {i}: "
        f"{accuracy * 100:.2f}%"
    )

print()

print(
    f"Mean Accuracy: "
    f"{mean_accuracy:.2f}%"
)

print(
    f"Std Accuracy: "
    f"{std_accuracy:.2f}%"
)


print()
print("SUBJECT-LEVEL GROUP VARIABILITY")

for group in [
    "HC",
    "MCI",
    "AD"
]:

    group_accuracies = [
        result["accuracy"]
        for result in subject_results
        if result["group"] == group
    ]

    print(
        f"{group}: "
        f"Mean = "
        f"{np.mean(group_accuracies) * 100:.2f}% | "
        f"Std = "
        f"{np.std(group_accuracies) * 100:.2f}%"
    )

print("=" * 70)
