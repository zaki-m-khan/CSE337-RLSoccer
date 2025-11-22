import numpy as np

class SarsaTileAgent:
    def __init__(self, n_features, actions, alpha, gamma, eps, seed=0):
        self.n_features = n_features
        self.actions = np.array(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        self.W = np.zeros((len(self.actions), n_features), dtype=np.float32)

    def select_action(self, f):
        if self.rng.random() < self.eps:
            return self.rng.choice(self.actions)
        q = self.q_values(f)
        return self.actions[int(np.argmax(q))]

    def q_values(self, f):
        return self.W[:, f].sum(axis=1)

    def update(self, f, a, r, f_next, a_next, done):
        q_next = np.max(self.W[:, f_next].sum(axis=1))
        td_target = r + (1 - done) * self.gamma * q_next
        td_error  = td_target - self.W[a, f].sum()
        self.W[a, f] += self.alpha * td_error

