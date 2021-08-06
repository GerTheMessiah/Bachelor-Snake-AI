import torch as T
import torch.nn as nn

from src.snakeAI.agents.ppo.actor_critic import ActorCritic
from src.snakeAI.agents.ppo.memoryPPO import Memory


class Agent:
    def __init__(self, lr_actor=1.0e-3, lr_critic=1.5e-3, gamma=0.99, K_epochs=10, eps_clip=0.2, gpu=False):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.critic_coeff = 0.5
        self.ent_coeff = 0.01
        self.device = T.device('cuda:0' if T.cuda.is_available() and gpu else 'cpu')
        self.model_id = 0
        self.MSEloss = nn.MSELoss()

        self.mem = Memory(device=self.device)
        self.policy = ActorCritic(n_actions=3, lr_actor=lr_actor, lr_critic=lr_critic, device=self.device)
        self.old_policy = ActorCritic(n_actions=3, lr_actor=lr_actor, lr_critic=lr_critic, device=self.device)
        self.old_policy.load_state_dict(self.policy.state_dict(destination=None))

    def learn(self):
        old_av, old_scalar, old_action, probs_old, reward_list, dones_list = self.mem.get_data()

        rewards = self.generate_reward(reward_list, dones_list)

        rewards = T.tensor(rewards, dtype=T.float64).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        for _ in range(self.K_epochs):
            probs, state_values, dist_entropy = self.policy.evaluate(old_av, old_scalar, old_action)

            ratios = T.exp(probs - probs_old)

            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = T.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss_actor = -(T.min(surr1, surr2) + dist_entropy * self.ent_coeff).mean()

            loss_critic = self.critic_coeff * self.MSEloss(rewards, state_values)
            loss = loss_actor + loss_critic

            self.policy.optimizer.zero_grad()
            loss.backward()
            # T.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())
        T.cuda.empty_cache()

    def generate_reward(self, rewards_in, terminals_in):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards_in), reversed(terminals_in)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        return rewards

    def load_model(self, state_dict):
        self.old_policy.load_state_dict(state_dict=state_dict)
        self.policy.load_state_dict(state_dict=state_dict)
