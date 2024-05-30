import numpy as np

class Learner:
    def __init__(self, inh_inds, A_d=1, A_l=1, tau_d=1, tau_l=1, a_d=0.1, a_l=-0.1): # A_d=0.8, A_l=0.25, tau_d=0.7, tau_l=1, a_d=0.5, a_l=-0.5
        self.inh_inds = inh_inds
        self.A_d = A_d
        self.A_l = A_l
        self.tau_d = tau_d
        self.tau_l = tau_l
        self.a_d = a_d
        self.a_l = a_l

    def W_d(self, s_d):
        return np.where(s_d > 0, self.A_d * np.exp(-s_d / self.tau_d), 0)

    def W_l(self, s_l):
        return np.where(s_l > 0, -self.A_l * np.exp(-s_l / self.tau_l), 0)

    def update(self, t, S_l_i, S_d_i, S_in, dt=1):
        # S_l [num_ts], S_d [num_ts], S_in [num_neurons, num_ts]

        dws = []
        num_neurons, num_ts = S_in.shape

        times = np.repeat(np.arange(401).reshape(1, -1), num_neurons, axis=0)

        dw_d = np.where(S_in, self.W_d(-times + t), 0).sum(1)
        dw_l = np.where(S_in, self.W_l(-times + t), 0).sum(1)

        mul = np.ones((num_neurons, ))
        mul[self.inh_inds] = -1

        dw = mul * (S_d_i * (self.a_d + dw_d) + S_l_i * (self.a_l + dw_l))

        return dw.reshape(-1, 1)
