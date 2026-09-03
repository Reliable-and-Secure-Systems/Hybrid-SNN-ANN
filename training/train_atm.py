import argparse

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from data.atm_dataset import load_atm_dataset
from encoding.atm_encoder import encode_atm_sample
from models.atm_hybrid_model import ATMHybridSNNANN


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--time-steps",
    type=int,
    default=20
)

args = parser.parse_args()

TIME_STEPS = args.time_steps

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

TEST_SIZE = 0.2
RANDOM_STATE = 42

INPUT_SIZE = 68 * 68
HIDDEN_SIZE = 128
NUM_CLASSES = 2


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# LOAD REAL ATM DATA
# ============================================================

X, y = load_atm_dataset()

print()
print("=" * 60)
print("REAL ATM DATASET")
print("=" * 60)

print("Data shape   :", X.shape)
print("Labels shape :", y.shape)
print("LEFT samples :", int(np.sum(y == 0)))
print("RIGHT samples:", int(np.sum(y == 1)))


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train_x_raw, test_x_raw, train_y, test_y = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print()
print("Training samples:", len(train_x_raw))
print("Testing samples :", len(test_x_raw))

print(
    "Training LEFT/RIGHT:",
    int(np.sum(train_y == 0)),
    int(np.sum(train_y == 1))
)

print(
    "Testing LEFT/RIGHT:",
    int(np.sum(test_y == 0)),
    int(np.sum(test_y == 1))
)


# ============================================================
# ENCODE DATA
# ============================================================

print()
print("=" * 60)
print("ENCODING")
print("=" * 60)

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


train_encoded = torch.from_numpy(
    np.array(train_encoded, dtype=np.float32)
)

test_encoded = torch.from_numpy(
    np.array(test_encoded, dtype=np.float32)
)

train_labels = torch.tensor(
    train_y,
    dtype=torch.long
)

test_labels = torch.tensor(
    test_y,
    dtype=torch.long
)


print("Training encoded shape:", train_encoded.shape)
print("Testing encoded shape :", test_encoded.shape)


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = TensorDataset(
    train_encoded,
    train_labels
)

test_dataset = TensorDataset(
    test_encoded,
    test_labels
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

model = ATMHybridSNNANN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        # batch × time × features
        # →
        # time × batch × features

        batch_x = batch_x.permute(1, 0, 2)

        optimizer.zero_grad()

        output, snn_features, spikes = model(
            batch_x
        )

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

    average_loss = (
        total_loss / len(train_loader)
    )

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Loss: {average_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )


# ============================================================
# TESTING
# ============================================================

print()
print("=" * 60)
print("TESTING")
print("=" * 60)

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for batch_x, batch_y in test_loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        batch_x = batch_x.permute(1, 0, 2)

        output, _, _ = model(
            batch_x
        )

        predictions = output.argmax(dim=1)

        correct += (
            predictions == batch_y
        ).sum().item()

        total += batch_y.size(0)


test_accuracy = (
    100 * correct / total
)

print(
    f"Test Accuracy: {test_accuracy:.2f}%"
)

print(
    f"Correct: {correct}/{total}"
)

print("=" * 60)


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "models/atm_hybrid_trained.pt"
)

print()
print("Model saved to:")
print("models/atm_hybrid_trained.pt")
