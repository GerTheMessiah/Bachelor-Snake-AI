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

    """
    Method for propagating input through network.
    @:param av: First part of observation -> shape (6x13x13).
    @:param scalar_obs: Second part of observation -> shape (1x41)
    @:returns policy: Probability distribution of all actions. value: Value of the Critic.
    """
    def forward(self, av, scalar_obs):
        policy = self.ACTOR(av, scalar_obs)
        value = self.CRITIC(av, scalar_obs)
        return policy, value

    """
    Method for propagating input through network.
    @:param av: First part of observation -> shape (6x13x13).
    @:param scalar_obs: Second part of observation -> shape (1x41)
    @:returns av: Tensor of av. scalar_obs: Tensor of scalar_obs. action: Returned action. log_prob: Logarithmic probability of the action
    """
    @T.no_grad()
    def act(self, av, scalar_obs):
        av = T.from_numpy(av).to(self.DEVICE)
        scalar_obs = T.from_numpy(scalar_obs).to(self.DEVICE)
        policy = self.ACTOR(av, scalar_obs)
        dist = Categorical(policy)
        action_tensor = dist.sample()
        action = action_tensor.item()

        return av, scalar_obs, action, dist.log_prob(action_tensor)

    """
    Method for propagating input through network.
    @:param av: First part of observation -> shape (6x13x13).
    @:param scalar_obs: Second part of observation -> shape (1x41)
    @:returns av: Tensor of av. scalar_obs: Tensor of scalar_obs. action: Returned action.
    """
    @T.no_grad()
    def act_test(self, av, scalar_obs):
        av = T.from_numpy(av).to(self.DEVICE)
        scalar_obs = T.from_numpy(scalar_obs).to(self.DEVICE)
        policy = self.ACTOR(av, scalar_obs)
        action = T.argmax(policy).item()
        return av, scalar_obs, action

    """
    Method for propagating input through network.
    @:param av: First part of observation -> shape (Xx6x13x13) | X = batch_size.
    @:param scalar_obs: Second part of observation -> shape (Xx1x41) | X = batch_size.
    @:returns action_probs: New logarithmic probability of action. value: Value of the critic. dist_entropy: Entropy of the policy.
    """
    def evaluate(self, av, scalar_obs, action):
        policy, value = self.forward(av, scalar_obs)
        dist = Categorical(policy)

        action_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        value = T.squeeze(value)
        return action_probs, value, dist_entropy
