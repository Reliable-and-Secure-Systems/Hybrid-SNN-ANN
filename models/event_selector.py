import torch
import torch.nn as nn


class EventSelector(nn.Module):

    def __init__(self, top_k=20):
        super().__init__()

        self.top_k = top_k

    def forward(self, features):

        # features represent average SNN firing activity.
        # Higher activity = more active event feature.

        activity = features

        # Number of features that actually contain activity
        active_features = (activity > 0).sum(dim=1)

        # Select the most active features
        k = min(
            self.top_k,
            features.size(1)
        )

        values, indices = torch.topk(
            activity,
            k,
            dim=1
        )

        return values, indices, active_features