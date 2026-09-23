import torch

from models.snn_model import SNNModel
from models.event_selector import EventSelector
from models.ann_model import ANNModel

# --------------------
# Models
# --------------------

snn = SNNModel()
selector = EventSelector(top_k=10)
ann = ANNModel()

# --------------------
# Dummy Input
# --------------------

dummy_input = torch.rand(1, 500)

# --------------------
# Forward Pass
# --------------------

features = snn(dummy_input)

# Handle models that return multiple values
if isinstance(features, tuple):
    features = features[0]

print("SNN Output:", features.shape)

selected = selector(features)

# EventSelector may return:
# (selected_features, selected_indices)
if isinstance(selected, tuple):
    selected = selected[0]

print("Selected Features:", selected.shape)

prediction = ann(selected)

print("Prediction:", prediction.shape)
print(prediction)
