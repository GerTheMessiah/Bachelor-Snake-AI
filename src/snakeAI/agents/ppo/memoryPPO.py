import torch as T


class Memory:
    def __init__(self, mem_size=2000, batch_size=64, device='cpu'):
        self.mem_size = mem_size
        self.counter = 0
        self.batch_size = batch_size
        self.av = T.zeros((self.mem_size, 6, 13, 13), dtype=T.float64, device=device)
        self.scalar_obs = T.zeros((self.mem_size, 41), dtype=T.float64, device=device)
        self.actions = T.zeros(self.mem_size, dtype=T.int64, device=device)
        self.probs = T.zeros(self.mem_size, dtype=T.float64, device=device)
        self.rewards = []
        self.dones = []

    def store(self, av, scalar_obs, action, probs, reward, done):
        self.av[self.counter, ...] = av.clone().detach()
        self.scalar_obs[self.counter, ...] = scalar_obs.clone().detach()
        self.actions[self.counter] = action
        self.probs[self.counter] = probs.clone().detach()
        self.rewards.append(reward)
        self.dones.append(done)
        self.counter += 1

    def get_data(self):
        return self.av[:self.counter, ...], \
               self.scalar_obs[:self.counter, ...], \
               self.actions[:self.counter, ...], \
               self.probs[:self.counter, ...], \
               self.rewards, \
               self.dones

    def __len__(self):
        return self.counter

    def clear_memory(self):
        del self.rewards[:]
        del self.dones[:]
        self.counter = 0
