import torch as T


class Memory:
    def __init__(self, MEM_SIZE=2500, AV_DIMS=(6, 13, 13), SCALAR_OBS_DIMS=41, DEVICE='cpu'):
        self.MEM_SIZE = MEM_SIZE
        self.counter = 0
        self.AV = T.zeros((self.MEM_SIZE, *AV_DIMS), dtype=T.float64, device=DEVICE)
        self.SCALAR_OBS = T.zeros((self.MEM_SIZE, SCALAR_OBS_DIMS), dtype=T.float64, device=DEVICE)
        self.ACTION = T.zeros(self.MEM_SIZE, dtype=T.int64, device=DEVICE)
        self.LOG_PROBABILITY = T.zeros(self.MEM_SIZE, dtype=T.float64, device=DEVICE)
        self.REWARD = []
        self.IS_TERMINAL = []

    def store(self, av, scalar_obs, action, log_probability, reward, is_terminal):
        self.AV[self.counter, ...] = av.clone().detach()
        self.SCALAR_OBS[self.counter, ...] = scalar_obs.clone().detach()
        self.ACTION[self.counter] = action
        self.LOG_PROBABILITY[self.counter] = log_probability.clone().detach()
        self.REWARD.append(reward)
        self.IS_TERMINAL.append(is_terminal)
        self.counter += 1

    def get_data(self):
        return self.AV[:self.counter, ...], \
               self.SCALAR_OBS[:self.counter, ...], \
               self.ACTION[:self.counter, ...], \
               self.LOG_PROBABILITY[:self.counter, ...], \
               self.REWARD, \
               self.IS_TERMINAL

    def __len__(self):
        return self.counter

    def clear_memory(self):
        del self.REWARD[:]
        del self.IS_TERMINAL[:]
        self.counter = 0
