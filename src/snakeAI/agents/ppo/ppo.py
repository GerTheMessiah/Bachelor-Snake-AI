import numpy as np
import torch as T
import torch.nn as nn

from src.snakeAI.agents.ppo.actor_critic import ActorCritic
from src.snakeAI.agents.ppo.memoryPPO import Memory


class Agent:
    def __init__(self, LR_ACTOR=1.5e-4, LR_CRITIC=3.0e-4, GAMMA=0.95, K_EPOCHS=10, EPS_CLIP=0.2, GPU=False):
        self.GAMMA = GAMMA
        self.EPS_CLIP = EPS_CLIP
        self.K_EPOCHS = K_EPOCHS
        self.CRITIC_COEFFICIENT = 0.5
        self.ENT_COEFFICIENT = 0.01
        self.DEVICE = T.device('cuda:0' if T.cuda.is_available() and GPU else 'cpu')
        self.LOSS = nn.MSELoss()

        self.MEM = Memory(DEVICE=self.DEVICE)
        self.POLICY = ActorCritic(N_ACTIONS=3, LR_ACTOR=LR_ACTOR, LR_CRITIC=LR_CRITIC, DEVICE=self.DEVICE)
        self.OLD_POLICY = ActorCritic(N_ACTIONS=3, LR_ACTOR=LR_ACTOR, LR_CRITIC=LR_CRITIC, DEVICE=self.DEVICE)
        self.OLD_POLICY.load_state_dict(self.POLICY.state_dict(destination=None))

    """
    Method for learning.
    """
    def learn(self):
        # If memory has not enough experiences.
        if self.MEM.counter < 64:
            return
        # Get data from memory.
        old_av, old_scalar, old_action, probs_old, reward_list, dones_list = self.MEM.get_data()
        # Discount rewards.
        rewards = self.generate_reward(reward_list, dones_list)
        rewards = T.tensor(rewards, dtype=T.float64, device=self.DEVICE)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # Generate batch indices.
        batch = np.random.randint(self.MEM.counter, size=(self.K_EPOCHS, self.MEM.counter))
        # Repeat K_EPOCHS times.
        for j in range(self.K_EPOCHS):
            old_av_b = old_av[batch[j, ...]]
            old_scalar_b = old_scalar[batch[j, ...]]
            old_action_b = old_action[batch[j, ...]]
            probs_old_b = probs_old[batch[j, ...]]
            rewards_b = rewards[batch[j, ...]]
            # Compute new logarithmic probability of action, values and entropy of policy.
            probs, state_values, dist_entropy = self.POLICY.evaluate(old_av_b, old_scalar_b, old_action_b)
            # Determine ratios.
            ratios = T.exp(probs - probs_old_b)
            # Determine advantages.
            advantages = rewards_b - state_values.detach()
            # Calculate surrogate losses.
            surr1 = ratios * advantages
            surr2 = T.clamp(ratios, 1 - self.EPS_CLIP, 1 + self.EPS_CLIP) * advantages
            # Determine actor loss.
            loss_actor = -(T.min(surr1, surr2) + dist_entropy * self.ENT_COEFFICIENT).mean()
            # Determine critic loss.
            loss_critic = self.CRITIC_COEFFICIENT * self.LOSS(rewards_b, state_values)
            # Combine losses.
            loss = loss_actor + loss_critic
            # Update network.
            self.POLICY.OPTIMIZER.zero_grad()
            loss.backward()
            self.POLICY.OPTIMIZER.step()
        # Update action determine policy with new updated policy.
        self.OLD_POLICY.load_state_dict(self.POLICY.state_dict())
        T.cuda.empty_cache()
        # Clear memory.
        self.MEM.clear_memory()

    """
    This method discounts the rewards.
    @:param rewards_in: Input rewards.
    @:param terminals_in: Input terminals. Is the state of obtained rewards a terminal one?
    """
    def generate_reward(self, reward_in, terminal_in):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(reward_in), reversed(terminal_in)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.GAMMA * discounted_reward
            rewards.insert(0, discounted_reward)
        return rewards

    def load_model(self, MODEL_PATH):
        try:
            state_dict = T.load(MODEL_PATH)
            self.OLD_POLICY.load_state_dict(state_dict=state_dict)
            self.POLICY.load_state_dict(state_dict=state_dict)
        except IOError:
            print("\nError while loading model.")

    def store_model(self, PATH):
        T.save(self.POLICY.state_dict(), PATH)
        save_worked = False
        while not save_worked:
            try:
                self.POLICY.load_state_dict(T.load(PATH))
                save_worked = True
            except FileNotFoundError:
                T.save(self.POLICY.state_dict(), PATH)
        print("\nmodel saved.")
