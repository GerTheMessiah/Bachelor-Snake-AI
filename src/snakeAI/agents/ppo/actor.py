import torch.nn as nn
import torch as T

from src.snakeAI.agents.common.av_net import AV_NET


class ActorNetwork(nn.Module):
    def __init__(self, OUTPUT=3, SCALAR_IN=41):
        super(ActorNetwork, self).__init__()
        self.AV_NET = AV_NET()

        self.ACTOR_TAIL = nn.Sequential(
            nn.Linear(128 + SCALAR_IN, 64),
            nn.ReLU(),
            nn.Linear(64, OUTPUT),
            nn.Softmax(dim=-1)
        )

    def forward(self, AV, SCALAR_OBS):
        av_out = self.AV_NET(AV)
        cat = T.cat((av_out, SCALAR_OBS), dim=-1)
        actor_out = self.ACTOR_TAIL(cat)
        return actor_out
