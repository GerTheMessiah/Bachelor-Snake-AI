import torch.nn as nn
import torch as T
from torch.optim.adam import Adam


class CriticNetwork(nn.Module):
    def __init__(self, scalar_in=41, lr=1.5e-3, device='cpu'):
        super(CriticNetwork, self).__init__()
        self.av_net = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(392, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.critic_head = nn.Sequential(
            nn.Linear(128 + scalar_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.device = device

    def forward(self, av, scalar_obs):
        av_out = self.av_net(av)
        cat = T.cat((av_out, scalar_obs), dim=-1)
        critic_out = self.critic_head(cat)
        return critic_out
