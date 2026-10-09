import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

from data.atm_dataset import load_atm_diagnostic_dataset
from encoding.atm_encoder import encode_atm_sample
from models.atm_hybrid_model import ATMHybridSNNANN


TIME_STEPS = 10
BETA = 0.8
HIDDEN_SIZE = 128
INPUT_SIZE = 68 * 68
NUM_CLASSES = 3

MODEL_PATH = "models/atm_hybrid_trained.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("Device:", DEVICE)

X, y, subjects, files = load_atm_diagnostic_dataset()

# Use the exact same subject split as training
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

_, test_idx = next(
    splitter.split(X, y, groups=subjects)
)

test_x = X[test_idx]
test_y = y[test_idx]
test_subjects = subjects[test_idx]


print()
print("=" * 60)
print("DIAGNOSTIC EVALUATION")
print("=" * 60)

print("Test samples:", len(test_x))
print("Test subjects:", sorted(np.unique(test_subjects)))

print(
    "HC/MCI/AD:",
    int(np.sum(test_y == 0)),
    int(np.sum(test_y == 1)),
    int(np.sum(test_y == 2))
)


print()
print("Encoding test data...")

encoded = []

for i, sample in enumerate(test_x):

    spikes = encode_atm_sample(
        sample,
        time_steps=TIME_STEPS,
        seed=10000 + i
    )

    encoded.append(spikes)


encoded = torch.from_numpy(
    np.array(encoded, dtype=np.float32)
)

labels = torch.tensor(
    test_y,
    dtype=torch.long
)


model = ATMHybridSNNANN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_classes=NUM_CLASSES,
    beta=BETA
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


predictions = []

with torch.no_grad():

    for i in range(0, len(encoded), 16):

        batch_x = encoded[i:i + 16].to(DEVICE)

        batch_x = batch_x.permute(1, 0, 2)

        output, _, _, _ = model(batch_x)

        batch_predictions = output.argmax(dim=1)

        predictions.extend(
            batch_predictions.cpu().numpy()
        )


predictions = np.array(predictions)


accuracy = accuracy_score(
    test_y,
    predictions
)

cm = confusion_matrix(
    test_y,
    predictions,
    labels=[0, 1, 2]
)


print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print(f"Overall accuracy: {accuracy * 100:.2f}%")

print()
print("Confusion Matrix")
print("Rows = Actual")
print("Columns = Predicted")
print()
print("             HC    MCI    AD")
print(f"HC       {cm[0,0]:6d} {cm[0,1]:6d} {cm[0,2]:6d}")
print(f"MCI      {cm[1,0]:6d} {cm[1,1]:6d} {cm[1,2]:6d}")
print(f"AD       {cm[2,0]:6d} {cm[2,1]:6d} {cm[2,2]:6d}")


print()
print("Classification Report")
print()

print(
    classification_report(
        test_y,
        predictions,
        labels=[0, 1, 2],
        target_names=["HC", "MCI", "AD"],
        zero_division=0
    )
)

print("=" * 60)
