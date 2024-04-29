import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, num_classes,action_length=3):
        """

        :param state_length: we give OHLC as input to the network
        :param action_length: Buy, Sell, Idle
        """
        super(Decoder, self).__init__()

        self.policy_network = nn.Sequential(
            nn.Linear(num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, action_length))

    def forward(self, x):
        output = self.policy_network(x).squeeze(1)
        return output
