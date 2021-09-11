import torch as T
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam

from src.snakeAI.agents.ppo.actor import ActorNetwork
from src.snakeAI.agents.ppo.critic import CriticNetwork


class ActorCritic(nn.Module):
    def __init__(self, N_ACTIONS=3, LR_ACTOR=1.5e-4, LR_CRITIC=3.0e-4, DEVICE="cpu"):
        super(ActorCritic, self).__init__()
        T.set_default_dtype(T.float64)
        T.manual_seed(10)
        self.ACTOR = ActorNetwork(OUTPUT=N_ACTIONS)
        self.CRITIC = CriticNetwork()
        self.OPTIMIZER = Adam([{'params': self.ACTOR.parameters(), 'lr': LR_ACTOR},
                               {'params': self.CRITIC.parameters(), 'lr': LR_CRITIC}])
        self.DEVICE = DEVICE
        self.to(self.DEVICE)

    def forward(self, av, scalar_obs):
        policy = self.ACTOR(av, scalar_obs)
        value = self.CRITIC(av, scalar_obs)
        return policy, value

    @T.no_grad()
    def act(self, av, scalar_obs):
        av = T.from_numpy(av).to(self.DEVICE)
        scalar_obs = T.from_numpy(scalar_obs).to(self.DEVICE)
        policy = self.ACTOR(av, scalar_obs)
        dist = Categorical(policy)
        action = dist.sample()
        return av, scalar_obs, action.item(), dist.log_prob(action)

    @T.no_grad()
    def act_test(self, av, scalar_obs):
        av = T.from_numpy(av).to(self.DEVICE)
        scalar_obs = T.from_numpy(scalar_obs).to(self.DEVICE)
        policy = self.ACTOR(av, scalar_obs)
        return av, scalar_obs, T.argmax(policy).item()

    def evaluate(self, av, scalar_obs, action):
        policy, value = self.forward(av, scalar_obs)
        dist = Categorical(policy)

        action_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_probs, T.squeeze(value), dist_entropy
