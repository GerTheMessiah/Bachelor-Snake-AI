import torch.nn as nn
import torch as T

from src.snakeAI.agents.common.av_net import AV_NET


class CriticNetwork(nn.Module):
    def __init__(self, SCALAR_IN=41):
        super(CriticNetwork, self).__init__()
        self.AV_NET = AV_NET()

        self.CRITIC_TAIL = nn.Sequential(
            nn.Linear(128 + SCALAR_IN, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    """
    Method for propagating input through network.
    @:param av: First part of observation -> shape (6x13x13).
    @:param scalar_obs: Second part of observation -> shape (1x41)
    @:return critic_out: Value of the critic.
    """
    def forward(self, av, scalar_obs):
        av_out = self.AV_NET(av)
        cat = T.cat((av_out, scalar_obs), dim=-1)
        critic_out = self.CRITIC_TAIL(cat)
        return critic_out
