import numpy as np

class Learner:
    def __init__(self, W, A_d=1, A_l=1, tau_d=1, tau_l=1, a_d=0, a_l=0):
        self.W = W
        self.A_d = A_d
        self.A_l = A_l
        self.tau_d = tau_d
        self.tau_l = tau_l
        self.a_d = a_d
        self.a_l = a_l

    def W_d(self, s_d):
        return np.where(s_d > 0, self.A_d * np.exp(-s_d / self.tau_d), 0)

    def W_l(self, s_l):
        return np.where(s_l > 0, - self.A_l * np.exp(-s_l / self.tau_l), 0)

    def update(self, t, S_l, S_d, S_in, times=None):
        # S_d: [n_neurons, times]
        new_times = np.tile(self.W_d(times), (4, 1))
        dw = (S_d[:, t] - S_l[:, t]) * (self.a_d + np.where(S_in, new_times, 0))
        return dw
