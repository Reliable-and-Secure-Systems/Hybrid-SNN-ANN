import torch
import torch.nn as nn


class EventSelector(nn.Module):

    def __init__(self, top_k=10):
        super().__init__()

        self.top_k = top_k

    def forward(self, features):

        values, indices = torch.topk(
            features,
            self.top_k,
            dim=1
        )

        return values