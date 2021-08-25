import torch as T
import numpy as np
from torch.nn import MSELoss

from src.snakeAI.agents.dqn.memoryDQN import Memory
from src.snakeAI.agents.dqn.q_net import QNetwork


class Agent:
    def __init__(self, LR=1e-4, N_ACTIONS=3, GAMMA=0.95, BATCH_SIZE=2 ** 5, EPS_START=1.0, EPS_END=0.01,
                 EPS_DEC=1e-5, MAX_MEM_SIZE=2 ** 11, GPU=True):
        self.gamma = GAMMA
        self.epsilon = EPS_START
        self.eps_min = EPS_END
        self.eps_dec = EPS_DEC
        self.lr = LR
        self.action_space = [i for i in range(N_ACTIONS)]
        self.mem_size = MAX_MEM_SIZE
        self.batch_size = BATCH_SIZE
        self.device = T.device('cuda:0' if T.cuda.is_available() and GPU else 'cpu')

        self.Q_eval = QNetwork(output=N_ACTIONS, lr=LR, device=self.device)
        self.loss = MSELoss()
        self.mem = Memory(max_mem_size=self.mem_size, in_dims_av=(6, 13, 13), in_dims_cat_obs=41,
                          batch_size=BATCH_SIZE, device=self.device)

    def act(self, av, scalar_obs):
        av = T.from_numpy(av).to(self.Q_eval.device)
        scalar_obs = T.from_numpy(scalar_obs).to(self.Q_eval.device)

        if np.random.random() > self.epsilon:
            with T.no_grad():
                actions = self.Q_eval(av, scalar_obs)
                action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return av, scalar_obs, action

    @T.no_grad()
    def act_test(self, av, scalar_obs):
        av = T.from_numpy(av).to(self.Q_eval.device)
        scalar_obs = T.from_numpy(scalar_obs).to(self.Q_eval.device)
        q_values = self.Q_eval(av, scalar_obs)
        return av, scalar_obs, T.argmax(q_values).item()

    def learn(self):
        if self.mem.mem_counter < self.batch_size:
            return

        max_mem = min(self.mem.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = T.arange(self.batch_size, dtype=T.long, device=self.device)
        av = self.mem.av[batch]
        scalar_obs = self.mem.scalar_obs[batch]
        av_ = self.mem.av_[batch]
        scalar_obs_ = self.mem.scalar_obs_[batch]
        actions = self.mem.actions[batch]
        terminal_batch = self.mem.terminals[batch]
        rewards_batch = self.mem.rewards[batch]

        q_eval = self.Q_eval(av, scalar_obs)[batch_index, actions]
        q_next = self.Q_eval(av_, scalar_obs_)
        q_next[terminal_batch] = 0.0
        q_target = rewards_batch + self.gamma * T.max(q_next, dim=1)[0]
        loss = self.loss(q_target, q_eval)
        self.Q_eval.optimizer.zero_grad()
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def store_model(self, path):
        path_model = path + "\\model"
        T.save(self.Q_eval.state_dict(), path_model)
        save_worked = False
        while not save_worked:
            try:
                self.Q_eval.load_state_dict(T.load(path_model))
                save_worked = True
            except FileNotFoundError:
                T.save(self.Q_eval.state_dict(), path_model)
        print("model saved.")

    def load_model(self, MODEL_PATH):
        try:
            state_dict = T.load(MODEL_PATH)
            self.Q_eval.load_state_dict(state_dict=state_dict)
        except IOError:
            print("Error while loading model.")
            return
