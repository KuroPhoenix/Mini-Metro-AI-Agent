import torch.nn as nn

class CNNPolicy(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 512), nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):          # x = (B,3,96,96) float32 [0,1]
        return self.net(x)
