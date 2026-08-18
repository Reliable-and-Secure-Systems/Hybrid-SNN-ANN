import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data.dummy_dataset import load_dummy_dataset
from encoding.rate_encoder import rate_encode

from models.snn_model import SNNModel
from models.ann_model import ANNModel
from models.event_selector import EventSelector

# ------------------------
# LOAD DATA
# ------------------------

eeg, labels = load_dummy_dataset()

# Use only first EEG channel

eeg = eeg[:,0,:]

# Rate Encode

spikes = np.array(
    [rate_encode(x) for x in eeg]
)

# Convert to Torch

X = torch.tensor(
    spikes,
    dtype=torch.float32
)

y = torch.tensor(
    labels,
    dtype=torch.long
)

print(X.shape)
print(y.shape)

# ------------------------
# MODELS
# ------------------------

snn = SNNModel()

selector = EventSelector(top_k=10)

ann = ANNModel()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(

    list(snn.parameters()) +

    list(ann.parameters()),

    lr=0.001

)

epochs = 10

for epoch in range(epochs):

    optimizer.zero_grad()

    features = snn(X)

    selected = selector(features)

    outputs = ann(selected)

    loss = criterion(outputs,y)

    loss.backward()

    optimizer.step()

    prediction = outputs.argmax(1)

    accuracy = (prediction==y).float().mean()

    print(

        f"Epoch {epoch+1}"

        f" | Loss {loss.item():.4f}"

        f" | Accuracy {accuracy.item()*100:.2f}%"

    )