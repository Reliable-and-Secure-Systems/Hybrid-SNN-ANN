import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class ATMHybridSNNANN(nn.Module):

    def __init__(
        self,
        input_size=4624,
        hidden_size=128,
        num_classes=2,
        beta=0.9
    ):

        super().__init__()

        spike_grad = surrogate.fast_sigmoid()

        # Store configuration for experiment tracking
        self.beta = beta

        # SNN part
        self.snn_fc = nn.Linear(
            input_size,
            hidden_size
        )

        self.snn_neuron = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad
        )

        # ANN part
        self.ann_fc1 = nn.Linear(
            hidden_size,
            64
        )

        self.ann_relu = nn.ReLU()

        self.ann_fc2 = nn.Linear(
            64,
            num_classes
        )

    def forward(self, x):

        # x shape:
        # time_steps × batch_size × input_size

        time_steps = x.size(0)

        mem = self.snn_neuron.init_leaky()

        spike_outputs = []

        for t in range(time_steps):

            current = self.snn_fc(x[t])

            spike, mem = self.snn_neuron(
                current,
                mem
            )

            spike_outputs.append(spike)

        # Combine SNN output over time
        spikes = torch.stack(spike_outputs)

        # Average firing activity
        snn_features = spikes.mean(dim=0)

        # First ANN stage
        ann_hidden = self.ann_relu(
            self.ann_fc1(snn_features)
        )

        # Final ANN classifier
        output = self.ann_fc2(ann_hidden)

        return output, snn_features, spikes, ann_hidden
