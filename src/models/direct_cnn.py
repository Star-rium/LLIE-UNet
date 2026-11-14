import torch
import torch.nn as nn

class DirectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # initialize CNN network
        # start with a 9 layers network (no pooling, downsampling)
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), # (3 channels) -> (32 channels, 3x3 kernel)
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1), # (32 channels -> 64 channels), kernel: 3x3
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, 3, padding=1), # (64 channels -> 32 channels), kernel: 3x3
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 3, 3, padding=1) # (32 channels -> 3 channels), kernel: 3x3
        )

    def forward(self, X):
        # output activation using sigmoid (assuming input is normalized)
        return torch.sigmoid(self.network(X))