import torch
import torch.nn as nn
import snntorch as snn

class SNNModel(nn.Module):

    def __init__(self, input_size=500, hidden_size=128, output_size=64):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)

        self.lif1 = snn.Leaky(beta=0.9)

        self.fc2 = nn.Linear(hidden_size, output_size)

        self.lif2 = snn.Leaky(beta=0.9)

    def forward(self, x):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk1, mem1 = self.lif1(self.fc1(x), mem1)

        spk2, mem2 = self.lif2(self.fc2(spk1), mem2)

        return spk2