import torch as T
import torch.nn as nn
from torch.distributions import Categorical

from src.snakeAI.agents.ppo.actor import ActorNetwork
from src.snakeAI.agents.ppo.critic import CriticNetwork


class ActorCritic(nn.Module):
    def __init__(self, n_actions=3, lr_actor=1.0e-3, lr_critic=1.5e-3, device="cpu"):
        super(ActorCritic, self).__init__()
        T.set_default_dtype(T.float64)
        T.manual_seed(10)
        self.actor = ActorNetwork(output=n_actions, lr=lr_actor, device=self.device)
        self.critic = CriticNetwork(lr=lr_critic, device=self.device)
        self.device = device
        self.to(self.device)

    def forward(self, av, scalar_obs):
        policy = self.actor(av, scalar_obs)
        value = self.critic(av, scalar_obs)
        return policy, value

    @T.no_grad()
    def act(self, av, scalar_obs):
        av = T.from_numpy(av).to(self.device)
        scalar_obs = T.from_numpy(scalar_obs).to(self.device)
        policy = self.actor(av, scalar_obs)

        dist = Categorical(policy)
        action = dist.sample()
        return av, scalar_obs, action.item(), dist.log_prob(action)

    def evaluate(self, av, scalar_obs, action):
        policy, value = self.forward(av, scalar_obs)
        dist = Categorical(policy)

        action_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_probs, T.squeeze(value), dist_entropy
