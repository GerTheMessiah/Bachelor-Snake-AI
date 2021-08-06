import torch.nn as nn
import torch as T
from torch.optim import Adam


class PrintSize(nn.Module):
    def __init__(self):
        super(PrintSize, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


class QNetwork(nn.Module):
    def __init__(self, output, lr, static_input=41, device="cuda:0"):
        super(QNetwork, self).__init__()
        T.set_default_dtype(T.float64)
        T.manual_seed(10)
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
            nn.Linear(256, 128),
        )

        self.Q_net = nn.Sequential(
            nn.Linear(128 + static_input, 64),
            nn.ReLU(),
            nn.Linear(64, output)
        )

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.device = device
        self.to(self.device)

    def forward(self, av, scalar_obs):
        av_out = self.av_net(av)
        cat = T.cat((av_out, scalar_obs), dim=-1)
        q_values = self.Q_net(cat)
        return q_values
