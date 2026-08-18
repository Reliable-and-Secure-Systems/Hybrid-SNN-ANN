import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class ATMHybridSNNANN(nn.Module):

    def __init__(self, input_size=4624, hidden_size=128, num_classes=2):

        super().__init__()

        spike_grad = surrogate.fast_sigmoid()

        # SNN part
        self.snn_fc = nn.Linear(input_size, hidden_size)

        self.snn_neuron = snn.Leaky(
            beta=0.9,
            spike_grad=spike_grad
        )

        # ANN part
        self.ann = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
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

        # ANN classification
        output = self.ann(snn_features)

        return output, snn_features, spikes