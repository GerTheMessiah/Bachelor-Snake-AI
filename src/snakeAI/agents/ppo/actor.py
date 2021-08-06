import torch.nn as nn
import torch as T


class ActorNetwork(nn.Module):
    def __init__(self, output=3, scalar_in=41):
        super(ActorNetwork, self).__init__()
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

        self.actor_head = nn.Sequential(
            nn.Linear(128 + scalar_in, 64),
            nn.ReLU(),
            nn.Linear(64, output),
            nn.Softmax(dim=-1)
        )

    def forward(self, av, scalar_obs):
        av_out = self.av_net(av)
        cat = T.cat((av_out, scalar_obs), dim=-1)
        actor_out = self.actor_head(cat)
        return actor_out
