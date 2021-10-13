import numpy as np
import torch as T


class Memory:
    def __init__(self, MAX_MEM_SIZE, AV_DIMENSION, SCALAR_OBS_DIMENSION, BATCH_SIZE=2 ** 6, DEVICE="cpu"):
        self.MEM_SIZE = MAX_MEM_SIZE
        self.counter = 0
        self.DEVICE = DEVICE
        self.BATCH_SIZE = BATCH_SIZE
        self.AV = T.zeros((self.MEM_SIZE, *AV_DIMENSION), dtype=T.float64, device=DEVICE)
        self.SCALAR_OBS = T.zeros((self.MEM_SIZE, SCALAR_OBS_DIMENSION), dtype=T.float64, device=DEVICE)
        self.ACTION = T.zeros(self.MEM_SIZE, dtype=T.long, device=DEVICE)
        self.REWARD = T.zeros(self.MEM_SIZE, dtype=T.float64, device=DEVICE)
        self.IS_TERMINAL = T.zeros(self.MEM_SIZE, dtype=T.bool, device=DEVICE)
        self.AV_ = T.zeros((self.MEM_SIZE, *AV_DIMENSION), dtype=T.float64, device=DEVICE)
        self.SCALAR_OBS_ = T.zeros((self.MEM_SIZE, SCALAR_OBS_DIMENSION), dtype=T.float64, device=DEVICE)

    """
    Method for saving data in memory.
    @:param av: First part of observation, around_view numpy array -> shape (6x13x13).
    @:param scalar_obs: Second part of observation, scalar_obs numpy array -> shape (1x41).
    @:param action: Determined action by the agent.
    @:param reward: Determine reward by environment.
    @:param terminal: Is the state terminal?
    @:param av_: around_view of the successor
    @:param scalar_obs_: scalar_obs of the successor.
    """
    def store(self, av, scalar_obs, action, reward, terminal, av_, scalar_obs_):
        index = self.counter % self.MEM_SIZE
        self.AV[index, ...] = av.clone().detach()
        self.SCALAR_OBS[index, ...] = scalar_obs.clone().detach()
        self.ACTION[index] = action
        self.REWARD[index] = reward
        self.IS_TERMINAL[index] = terminal
        self.AV_[index, ...] = T.tensor(av_, dtype=T.float64, device=self.DEVICE)
        self.SCALAR_OBS_[index, ...] = T.tensor(scalar_obs_, dtype=T.float64, device=self.DEVICE)
        self.counter += 1

    """
    Method for getting a random shuffled data batch.
    @:param manuel_batch_size: Manuel batch size
    """
    def get_data(self, manuel_batch_size=None):
        max_mem = min(self.counter, self.MEM_SIZE)
        size = self.BATCH_SIZE if not bool(manuel_batch_size) else manuel_batch_size
        batch = np.random.choice(max_mem, size, replace=False)

        batch_index = T.arange(size, dtype=T.long, device=self.DEVICE)
        return self.AV[batch], self.SCALAR_OBS[batch], self.ACTION[batch], self.REWARD[batch], \
               self.IS_TERMINAL[batch], self.AV_[batch], self.SCALAR_OBS_[batch], batch_index
