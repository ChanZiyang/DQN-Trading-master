import torch
import torch.nn as nn
import torch.nn.functional as F
from .iTransformer import Model

class Encoder(nn.Module):

    def __init__(self, config):
        """

        :param state_size: we give OHLC as input to the network
        :param action_length: Buy, Sell, Idle
        """
        super(Encoder, self).__init__()
        self.config = config
        self.encoder = Model(self.config)

    def forward(self, x):
        x = self.encoder(x)
        return x
