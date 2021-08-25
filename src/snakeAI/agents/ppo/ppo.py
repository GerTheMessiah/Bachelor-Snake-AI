import numpy as np
import torch as T
import torch.nn as nn

from src.snakeAI.agents.ppo.actor_critic import ActorCritic
from src.snakeAI.agents.ppo.memoryPPO import Memory


class Agent:
    def __init__(self, LR_ACTOR=1.5e-4, LR_CRITIC=3.0e-4, GAMMA=0.95, K_EPOCHS=10, EPS_CLIP=0.2, GPU=False):
        self.gamma = GAMMA
        self.eps_clip = EPS_CLIP
        self.K_epochs = K_EPOCHS
        self.critic_coeff = 0.5
        self.ent_coeff = 0.01
        self.device = T.device('cuda:0' if T.cuda.is_available() and GPU else 'cpu')
        self.MSEloss = nn.MSELoss()

        self.mem = Memory(device=self.device)
        self.policy = ActorCritic(n_actions=3, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, device=self.device)
        self.old_policy = ActorCritic(n_actions=3, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, device=self.device)
        self.old_policy.load_state_dict(self.policy.state_dict(destination=None))

    def learn(self):
        if self.mem.counter < self.mem.batch_size:
            return

        old_av, old_scalar, old_action, probs_old, reward_list, dones_list = self.mem.get_data()

        rewards = self.generate_reward(reward_list, dones_list)

        rewards = T.tensor(rewards, dtype=T.float64, device=self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        batch = np.random.randint(self.mem.counter, size=(self.K_epochs, self.mem.counter))

        for j in range(self.K_epochs):
            old_av_b = old_av[batch[j, ...]]
            old_scalar_b = old_scalar[batch[j, ...]]
            old_action_b = old_action[batch[j, ...]]
            probs_old_b = probs_old[batch[j, ...]]
            rewards_b = rewards[batch[j, ...]]

            probs, state_values, dist_entropy = self.policy.evaluate(old_av_b, old_scalar_b, old_action_b)

            ratios = T.exp(probs - probs_old_b)

            advantages = rewards_b - state_values.detach()

            surr1 = ratios * advantages
            surr2 = T.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss_actor = -(T.min(surr1, surr2) + dist_entropy * self.ent_coeff).mean()

            loss_critic = self.critic_coeff * self.MSEloss(rewards_b, state_values)
            loss = loss_actor + loss_critic

            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())
        T.cuda.empty_cache()
        self.mem.clear_memory()

    def generate_reward(self, rewards_in, terminals_in):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards_in), reversed(terminals_in)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        return rewards

    def load_model(self, MODEL_PATH):
        try:
            state_dict = T.load(MODEL_PATH)
            self.old_policy.load_state_dict(state_dict=state_dict)
            self.policy.load_state_dict(state_dict=state_dict)
        except IOError:
            print("\nError while loading model.")

    def store_model(self, path):
        path_model = path
        T.save(self.policy.state_dict(), path_model)
        save_worked = False
        while not save_worked:
            try:
                self.policy.load_state_dict(T.load(path_model))
                save_worked = True
            except FileNotFoundError:
                T.save(self.policy.state_dict(), path_model)
        print("\nmodel saved.")
