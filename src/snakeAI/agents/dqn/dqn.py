import torch as T
import numpy as np
from torch.nn import MSELoss

from src.snakeAI.agents.dqn.memoryDQN import Memory
from src.snakeAI.agents.dqn.q_net import QNetwork


class Agent:
    def __init__(self, LR=1e-4, N_ACTIONS=3, GAMMA=0.95, BATCH_SIZE=2 ** 5, EPS_START=1.0, EPS_END=0.01,
                 EPS_DEC=1e-5, MAX_MEM_SIZE=2 ** 11, GPU=True):
        # constants
        self.GAMMA = GAMMA
        self.EPSILON = EPS_START
        self.EPS_MIN = EPS_END
        self.EPS_DEC = EPS_DEC
        self.LR = LR
        self.ACTION_SPACE = [i for i in range(N_ACTIONS)]
        self.MEM_SIZE = MAX_MEM_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.DEVICE = T.device('cuda:0' if T.cuda.is_available() and GPU else 'cpu')
        # initialization of the network.
        self.Q_NET = QNetwork(OUTPUT=N_ACTIONS, LR=LR, DEVICE=self.DEVICE)
        # initialization of the loss function.
        self.LOSS = MSELoss()
        # initialization of the memory.
        self.MEM = Memory(MAX_MEM_SIZE=self.MEM_SIZE, AV_DIMENSION=(6, 13, 13), SCALAR_OBS_DIMENSION=41,
                          BATCH_SIZE=BATCH_SIZE, DEVICE=self.DEVICE)
    """
    This method determines the action for training runs.
    @:param av: First part of observation, around_view numpy array -> shape (6x13x13).
    @:param scalar_obs: Second part of observation, scalar_obs numpy array -> shape (1x41)
    """
    def act(self, av, scalar_obs):
        av = T.from_numpy(av).to(self.Q_NET.DEVICE)
        scalar_obs = T.from_numpy(scalar_obs).to(self.Q_NET.DEVICE)

        if np.random.random() > self.EPSILON:
            with T.no_grad():
                actions = self.Q_NET(av, scalar_obs)
                action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.ACTION_SPACE)
        return av, scalar_obs, action

    """
    This method determines the action for test runs.
    @:param av: First part of observation, around_view numpy array -> shape (6x13x13).
    @:param scalar_obs: Second part of observation, scalar_obs numpy array -> shape (1x41)
    """
    @T.no_grad()
    def act_test(self, av, scalar_obs):
        av = T.from_numpy(av).to(self.Q_NET.DEVICE)
        scalar_obs = T.from_numpy(scalar_obs).to(self.Q_NET.DEVICE)
        q_values = self.Q_NET(av, scalar_obs)
        return av, scalar_obs, T.argmax(q_values).item()

    """
    This method represents the training routine.
    """
    def learn(self):
        if self.MEM.counter < self.BATCH_SIZE:
            return
        # get mini-batch for learning.
        av, scalar_obs, actions, rewards, is_terminal, av_, scalar_obs_, batch_index = self.MEM.get_data()
        # determine q-value of the action saved in memory.
        q_eval = self.Q_NET(av, scalar_obs)[batch_index, actions]
        # determine q-value of all action of the successor state.
        q_next = self.Q_NET(av_, scalar_obs_)
        # q-values of terminal stats are by definition zero.
        q_next[is_terminal] = 0.0
        # determine q-eval with the successor state. Q-Eval and q-target are in theory equal.
        q_target = rewards + self.GAMMA * T.max(q_next, dim=1)[0]
        # create loss.
        loss = self.LOSS(q_target, q_eval)
        # update network.
        self.Q_NET.OPTIMIZER.zero_grad()
        loss.backward()
        self.Q_NET.OPTIMIZER.step()
        # reduce EPSILON.
        self.EPSILON = self.EPSILON - self.EPS_DEC if self.EPSILON > self.EPS_MIN else self.EPS_MIN

    """
    This method saves the network.
    @:param PATH: Path for saving the network.
    """
    def store_model(self, PATH):
        T.save(self.Q_NET.state_dict(), PATH)
        save_worked = False
        while not save_worked:
            try:
                self.Q_NET.load_state_dict(T.load(PATH))
                save_worked = True
            except FileNotFoundError:
                T.save(self.Q_NET.state_dict(), PATH)
        print("model saved.")

    """
    This method loads the network.
    @:param PATH: Path for loading the network.
    """
    def load_model(self, PATH):
        try:
            state_dict = T.load(PATH)
            self.Q_NET.load_state_dict(state_dict=state_dict)
        except IOError:
            print("Error while loading model.")
            return
