import torch.nn as nn
import torch as T
from torch.optim import Adam

from src.snakeAI.agents.common.av_net import AV_NET


class PrintSize(nn.Module):
    def __init__(self):
        super(PrintSize, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


class QNetwork(nn.Module):
    def __init__(self, OUTPUT, LR, SCALAR_INPUT=41, DEVICE="cpu"):
        super(QNetwork, self).__init__()
        T.set_default_dtype(T.float64)
        T.manual_seed(10)
        self.AV_NET = AV_NET()

        self.Q_net = nn.Sequential(
            nn.Linear(128 + SCALAR_INPUT, 64),
            nn.ReLU(),
            nn.Linear(64, OUTPUT)
        )

        self.OPTIMIZER = Adam(self.parameters(), lr=LR)
        self.DEVICE = DEVICE
        self.to(self.DEVICE)

    """
    Method for propagating input through network.
    @:param av: First part of observation -> shape (6x13x13).
    @:param scalar_obs: Second part of observation -> shape (1x41)
    @:return q_values: Q-Values
    """
    def forward(self, av, scalar_obs):
        av_out = self.AV_NET(av)
        cat = T.cat((av_out, scalar_obs), dim=-1)
        q_values = self.Q_net(cat)
        return q_values
