import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from data.atm_dataset import load_atm_dataset
from encoding.atm_encoder import encode_atm_sample
from models.atm_hybrid_model import ATMHybridSNNANN


TIME_STEPS = 20
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


X, y = load_atm_dataset()

print("Original data shape:", X.shape)
print("Labels shape:", y.shape)


encoded_data = []

for i in range(len(X)):

    spikes = encode_atm_sample(
        X[i],
        time_steps=TIME_STEPS,
        seed=42 + i
    )

    encoded_data.append(spikes)


encoded_data = torch.from_numpy(np.array(encoded_data, dtype=np.float32))

labels = torch.tensor(
    y,
    dtype=torch.long
)


print("Encoded data shape:", encoded_data.shape)
print("Labels shape:", labels.shape)


from sklearn.model_selection import train_test_split


train_x, test_x, train_y, test_y = train_test_split(
    encoded_data,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("Training samples:", len(train_x))
print("Testing samples :", len(test_x))

print(
    "Training LEFT/RIGHT:",
    int((train_y == 0).sum()),
    int((train_y == 1).sum())
)

print(
    "Testing LEFT/RIGHT:",
    int((test_y == 0).sum()),
    int((test_y == 1).sum())
)

train_dataset = TensorDataset(
    train_x,
    train_y
)

test_dataset = TensorDataset(
    test_x,
    test_y
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



model = ATMHybridSNNANN(
    input_size=4624,
    hidden_size=128,
    num_classes=2
)

model = model.to(DEVICE)



criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


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

        # DataLoader gives:
        # batch × time × features

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        # Convert to:
        # time × batch × features

        batch_x = batch_x.permute(
            1, 0, 2
        )

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

    average_loss = (
        total_loss / len(train_loader)
    )

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Loss: {average_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )


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

        batch_x = batch_x.permute(
            1, 0, 2
        )

        output, _, _ = model(
            batch_x
        )

        predictions = output.argmax(
            dim=1
        )

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

print("=" * 60)

torch.save(
    model.state_dict(),
    "models/atm_hybrid_trained.pt"
)

print("Trained model saved to:")
print("models/atm_hybrid_trained.pt")