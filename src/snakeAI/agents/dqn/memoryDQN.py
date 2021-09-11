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

    def get_data(self, returned_data=None):
        max_mem = min(self.counter, self.MEM_SIZE)
        size = self.BATCH_SIZE if not bool(returned_data) else returned_data
        batch = np.random.choice(max_mem, size, replace=False)

        batch_index = T.arange(size, dtype=T.long, device=self.DEVICE)
        return self.AV[batch], self.SCALAR_OBS[batch], self.ACTION[batch], self.REWARD[batch], \
               self.IS_TERMINAL[batch], self.AV_[batch], self.SCALAR_OBS_[batch], batch_index
