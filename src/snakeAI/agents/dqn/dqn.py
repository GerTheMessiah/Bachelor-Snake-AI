import torch as T
import numpy as np
from torch.nn import MSELoss

from src.snakeAI.agents.common.utils import file_path
from src.snakeAI.agents.dqn.memoryDQN import Memory
from src.snakeAI.agents.dqn.q_net import QNetwork


class Agent:
    def __init__(self, lr=1e-4, n_actions=3, gamma=0.95, batch_size=2**5, epsilon_start=1.0, eps_end=0.01, eps_dec=1e-5, max_mem_size=2**11):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size

        self.Q_eval = QNetwork(output=n_actions, lr=lr, device="cuda:0")
        self.loss = MSELoss()
        self.mem = Memory(max_mem_size=self.mem_size, in_dims_av=(6, 13, 13), in_dims_cat_obs=(41,), batch_size=batch_size)

    def act(self, av, scalar_obs):
        if np.random.random() > self.epsilon:
            with T.no_grad():
                av = T.as_tensor(av, dtype=T.float64, device=self.Q_eval.device)
                scalar_obs = T.as_tensor(scalar_obs, dtype=T.float64, device=self.Q_eval.device)
                actions = self.Q_eval(av, scalar_obs)
                action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.mem.mem_counter < self.batch_size:
            return

        max_mem = min(self.mem.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = T.arange(self.batch_size, dtype=T.long)
        avs = self.mem.around_views[batch].to(self.Q_eval.device)
        cat_obs = self.mem.cat_obs[batch].to(self.Q_eval.device)
        new_avs = self.mem.new_around_views[batch].to(self.Q_eval.device)
        new_cat_obs = self.mem.new_cat_obs[batch].to(self.Q_eval.device)
        actions = self.mem.actions[batch].to(self.Q_eval.device)
        terminal_batch = self.mem.terminals[batch].to(self.Q_eval.device)
        rewards_batch = self.mem.rewards[batch].to(self.Q_eval.device)

        q_eval = self.Q_eval(avs, cat_obs)[batch_index, actions]
        q_next = self.Q_eval(new_avs, new_cat_obs)
        q_next[terminal_batch] = 0.0
        q_target = rewards_batch + self.gamma * T.max(q_next, dim=1)[0]
        loss = self.loss(q_target, q_eval)
        self.Q_eval.optimizer.zero_grad()
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.Q_eval.parameters(), 0.5)
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_model(self, new_save):
        T.save(self.Q_eval.state_dict(), file_path(dir=r"models\dqn_models", new_save=new_save, file_name="model"))

    def load_model(self, path=None):
        try:
            self.Q_eval.load_state_dict(T.load(file_path(dir=r"models\dqn_models", new_save=False, file_name="model")))
            if path is not None:
                self.Q_eval.load_state_dict(T.load(path))
        except IOError:
            print("Error while loading model.")
            return
