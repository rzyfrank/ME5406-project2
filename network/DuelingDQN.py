import torch
import torch.nn as nn
import numpy as np

CHANNEL=64

class DuelingDQNet(nn.Module):
    # VGG11 backbone
    def __init__(self, in_channel, num_action, num_data):
        super(DuelingDQNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, CHANNEL, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #112
            nn.Conv2d(CHANNEL, 2*CHANNEL, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #56
            nn.Conv2d(2*CHANNEL,4*CHANNEL, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(4*CHANNEL, 4*CHANNEL, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #28
            nn.Conv2d(4*CHANNEL, 8*CHANNEL, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(8*CHANNEL, 8*CHANNEL, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #14
            nn.Conv2d(8 * CHANNEL, 8 * CHANNEL, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(8 * CHANNEL, 8 * CHANNEL, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #7
            nn.Flatten()
        )
        self.fc_data = nn.Sequential(
            nn.Linear(num_data, 12), nn.ReLU()
        )

        self.fc_value = nn.Sequential(
            nn.Linear(512*7*7+12, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1)
        )


        self.fc_action = nn.Sequential(
            nn.Linear(512 * 7 * 7 + 12, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_action)
        )


    def forward(self, image, data):
        image = self.conv(image)
        data = self.fc_data(data)
        x = torch.cat([image, data], dim=1)
        values = self.fc_value(x)
        advantages = self.fc_action(x)
        Q = values + (advantages - advantages.mean())
        return Q