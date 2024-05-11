import numpy as np

class Learner:
    def __init__(self,A_d=1, A_l=1, tau_d=1, tau_l=1, a_d=0, a_l=0):
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


    def update(self, t, S_l, S_d, S_in, dt=1):
        # S_l [num_ts], S_d [num_ts], S_in [num_neurons, num_ts]

        num_neurons, num_ts = S_in.shape
        time_id = int(t / dt)

        dws = []
        for neuron in range(num_neurons):
            dw_d = dw_l = 0

            for j in range(len(S_d)):
                s_d = t - 0.001 * dt * j
                dw_d += self.W_d(s_d) * S_in[neuron, j]
            
            for j in range(len(S_l)):
                s_l = t - 0.001 * dt * j
                dw_l += self.W_l(s_l) * S_in[neuron, j]

            dws.append(S_d[time_id] * (self.a_d + dw_d) + S_l[time_id] * (self.a_l + dw_l))

        return dws