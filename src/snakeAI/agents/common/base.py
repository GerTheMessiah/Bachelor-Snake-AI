import torch.nn as nn
import torch as T
from torch.optim import Adam


class PrintSize(nn.Module):
    def __init__(self):
        super(PrintSize, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


class BaseNet(nn.Module):
    def __init__(self, output, head_type, lr, static_input=41, device="cuda:0"):
        super(BaseNet, self).__init__()
        T.set_default_dtype(T.float64)
        T.manual_seed(5)
        self.base_net = nn.Sequential(
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

        if head_type == "actor":
            self.head = nn.Sequential(
                nn.Linear(128 + static_input, 64),
                nn.ReLU(),
                nn.Linear(64, output),
                nn.Softmax(dim=-1)
            )

        else:
            self.head = nn.Sequential(
                nn.Linear(128 + static_input, 64),
                nn.ReLU(),
                nn.Linear(64, output)
            )

        self.device = device
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, av, scalar_obs):
        base_out = self.base_net(av)
        cat = T.cat((base_out, scalar_obs), dim=-1)
        res = self.head(cat)
        return res
