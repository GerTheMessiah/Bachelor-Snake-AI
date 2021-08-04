import torch as T


class Memory:
    def __init__(self, max_mem_size, in_dims_av, in_dims_cat_obs, batch_size=2 ** 6):
        self.mem_size = max_mem_size
        self.mem_counter = 0
        self.batch_size = batch_size
        self.around_views = T.zeros((self.mem_size, *in_dims_av), dtype=T.float64)
        self.cat_obs = T.zeros((self.mem_size, *in_dims_cat_obs), dtype=T.float64)
        self.actions = T.zeros(self.mem_size, dtype=T.long)
        self.rewards = T.zeros(self.mem_size, dtype=T.float64)
        self.terminals = T.zeros(self.mem_size, dtype=T.bool)
        self.new_around_views = T.zeros((self.mem_size, *in_dims_av), dtype=T.float64)
        self.new_cat_obs = T.zeros((self.mem_size, *in_dims_cat_obs), dtype=T.float64)

    def add(self, av, cat_obs, action, reward, done, new_av, new_cat_obs):
        index = self.mem_counter % self.mem_size
        self.around_views[index, ...] = T.tensor(av, dtype=T.float64)
        self.cat_obs[index, ...] = T.tensor(cat_obs, dtype=T.float64)
        self.actions[index] = action
        self.rewards[index] = reward
        self.terminals[index] = done
        self.new_around_views[index, ...] = T.tensor(new_av, dtype=T.float64)
        self.new_cat_obs[index, ...] = T.tensor(new_cat_obs, dtype=T.float64)
        self.mem_counter += 1
