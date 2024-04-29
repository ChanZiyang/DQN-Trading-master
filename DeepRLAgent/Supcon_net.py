import torch
import torch.nn as nn
import torch.nn.functional as F


class Supcon_net(nn.Module):

    def __init__(self, state_size):
        """

        :param state_size: we give OHLC as input to the network
        :param action_length: Buy, Sell, Idle
        """
        super(Supcon_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, state_size)
            )

    def forward(self, x):
        x = self.net(x)
        return x
