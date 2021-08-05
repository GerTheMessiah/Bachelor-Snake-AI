

class Memory:
    def __init__(self):
        self.av = []
        self.scalar_obs = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.dones = []

    def store(self, av, scalar_obs, action, probs, reward, done):
        self.av.append(av)
        self.scalar_obs.append(scalar_obs)
        self.actions.append(action)
        self.probs.append(probs)
        self.rewards.append(reward)
        self.dones.append(done)

    def generate_batches(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.av)

    def clear_memory(self):
        del self.av[:]
        del self.scalar_obs[:]
        del self.actions[:]
        del self.probs[:]
        del self.rewards[:]
        del self.dones[:]
