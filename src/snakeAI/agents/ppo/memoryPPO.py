import torch as T


class Memory:
    def __init__(self, MEM_SIZE=2500, AV_DIMS=(6, 13, 13), SCALAR_OBS_DIMS=41, DEVICE='cpu'):
        self.MEM_SIZE = MEM_SIZE
        self.counter = 0
        self.AV = T.zeros((self.MEM_SIZE, *AV_DIMS), dtype=T.float64, device=DEVICE)
        self.SCALAR_OBS = T.zeros((self.MEM_SIZE, SCALAR_OBS_DIMS), dtype=T.float64, device=DEVICE)
        self.ACTION = T.zeros(self.MEM_SIZE, dtype=T.int64, device=DEVICE)
        self.LOG_PROBABILITY = T.zeros(self.MEM_SIZE, dtype=T.float64, device=DEVICE)
        self.REWARD = list()
        self.IS_TERMINAL = list()

    """
    Method for saving data in memory.
    @:param av: First part of observation, around_view numpy array -> shape (6x13x13).
    @:param scalar_obs: Second part of observation, scalar_obs numpy array -> shape (1x41).
    @:param action: Determined action by the agent.
    @:param log_probability: Probability distribution of all actions.
    @:param reward: Determine reward by environment.
    @:param is_terminal: Is the state terminal?
    @:param av_: around_view of the successor.
    """
    def store(self, av, scalar_obs, action, log_probability, reward, is_terminal):
        self.AV[self.counter, ...] = av.clone().detach()
        self.SCALAR_OBS[self.counter, ...] = scalar_obs.clone().detach()
        self.ACTION[self.counter] = action
        self.LOG_PROBABILITY[self.counter] = log_probability.clone().detach()
        self.REWARD.append(reward)
        self.IS_TERMINAL.append(is_terminal)
        self.counter += 1
    """
    Method for getting all data.
    """
    def get_data(self):
        return self.AV[:self.counter, ...], \
               self.SCALAR_OBS[:self.counter, ...], \
               self.ACTION[:self.counter, ...], \
               self.LOG_PROBABILITY[:self.counter, ...], \
               self.REWARD, \
               self.IS_TERMINAL

    def __len__(self):
        return self.counter

    """
    Method for clearing the memory.
    """
    def clear_memory(self):
        del self.REWARD[:]
        del self.IS_TERMINAL[:]
        self.counter = 0
