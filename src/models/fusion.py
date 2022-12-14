import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(102, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 51),
        )

    def forward(self, x):
        x = torch.tensor(x)
        self.logits = self.linear_relu_stack(x.type(torch.FloatTensor).cuda())
        return self.logits