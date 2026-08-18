import torch

from models.snn_model import SNNModel
from models.event_selector import EventSelector

# Models
snn = SNNModel()
selector = EventSelector(top_k=10)

# Dummy EEG
dummy_input = torch.rand(1,500)

# Edge Processing
features = snn(dummy_input)

selected = selector(features)

print("Compressed Events Shape:", selected.shape)

# Save compressed events
torch.save(selected, "compressed_events.pt")

print("Compressed events saved successfully!")