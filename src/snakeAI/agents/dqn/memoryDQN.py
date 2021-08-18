import torch as T


class Memory:
    def __init__(self, max_mem_size, in_dims_av, in_dims_cat_obs, batch_size=2**6, device="cuda:0"):
        self.mem_size = max_mem_size
        self.mem_counter = 0
        self.batch_size = batch_size
        self.device = device
        self.av = T.zeros((self.mem_size, *in_dims_av), dtype=T.float64, device=device)
        self.scalar_obs = T.zeros((self.mem_size, in_dims_cat_obs), dtype=T.float64, device=device)
        self.actions = T.zeros(self.mem_size, dtype=T.long, device=device)
        self.rewards = T.zeros(self.mem_size, dtype=T.float64, device=device)
        self.terminals = T.zeros(self.mem_size, dtype=T.bool, device=device)
        self.av_ = T.zeros((self.mem_size, *in_dims_av), dtype=T.float64, device=device)
        self.scalar_obs_ = T.zeros((self.mem_size, in_dims_cat_obs), dtype=T.float64, device=device)

    def add(self, av, scalar_obs, action, reward, done, av_, scalar_obs_):
        index = self.mem_counter % self.mem_size
        self.av[index, ...] = av.clone().detach()
        self.scalar_obs[index, ...] = scalar_obs.clone().detach()
        self.actions[index] = action
        self.rewards[index] = reward
        self.terminals[index] = done
        self.av_[index, ...] = T.tensor(av_, dtype=T.float64, device=self.device)
        self.scalar_obs_[index, ...] = T.tensor(scalar_obs_, dtype=T.float64, device=self.device)
        self.mem_counter += 1

    def add_multiple(self, av, cat_obs, action, reward, done, new_av, new_cat_obs):
        length = action.size()
        index = self.mem_counter % self.mem_size
        if length + index < self.mem_size:
            self.av[index:length, ...] = av
            self.scalar_obs[index:length, ...] = cat_obs
            self.actions[index:length] = action
            self.rewards[index:length] = reward
            self.terminals[index:length] = done
            self.av_[index:length, ...] = new_av
            self.scalar_obs_[index:length, ...] = new_cat_obs
            self.mem_counter += length
        else:
            fill_len = self.mem_size - index
            self.av[index:fill_len, ...] = av[:fill_len]
            self.scalar_obs[index:fill_len, ...] = cat_obs[:fill_len]
            self.actions[index:fill_len] = action
            self.rewards[index:fill_len] = reward
            self.terminals[index:fill_len] = done
            self.av_[index:fill_len, ...] = new_av[:fill_len]
            self.scalar_obs_[index:fill_len, ...] = new_cat_obs[:fill_len]
            self.mem_counter += length

            index = self.mem_counter % self.mem_size
            self.av[:fill_len, ...] = av[fill_len:index]
            self.scalar_obs[:fill_len, ...] = cat_obs[fill_len:index]
            self.actions[:fill_len] = action
            self.rewards[:fill_len] = reward
            self.terminals[:fill_len] = done
            self.av_[:fill_len, ...] = new_av[fill_len:index]
            self.scalar_obs_[:fill_len, ...] = new_cat_obs[fill_len:index]


class SplitMemory:
    def __init__(self, max_mem_size, in_dims_av, in_dims_cat_obs, batch_size=2 ** 6, threshold=-6.0, ):
        self.mem_size = max_mem_size
        self.mem_counter = 0
        self.batch_size = batch_size
        self.threshold = threshold

        self.mem1 = Memory(max_mem_size // 2, in_dims_av, in_dims_cat_obs, batch_size=2 ** 6)
        self.mem2 = Memory(max_mem_size // 2, in_dims_av, in_dims_cat_obs, batch_size=2 ** 6)
        self.tmp_mem = Memory(max_mem_size, in_dims_av, in_dims_cat_obs, batch_size=2 ** 6)

    def store(self, av, cat_obs, action, reward, done, new_av, new_cat_obs):
        if not done:
            self.tmp_mem.add(av, cat_obs, action, reward, done, new_av, new_cat_obs)
        else:
            self.tmp_mem.add(av, cat_obs, action, reward, done, new_av, new_cat_obs)
            sum_rewards = T.sum(self.tmp_mem.rewards[:self.tmp_mem.mem_counter]).item()
            if sum_rewards > self.threshold:
                self.mem1.add_multiple(av, cat_obs, action, reward, done, new_av, new_cat_obs)
            else:
                self.mem2.add_multiple(av, cat_obs, action, reward, done, new_av, new_cat_obs)
