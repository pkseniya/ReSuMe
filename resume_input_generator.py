import numpy as np

def poisson_generator(N, dt, nt, rate):
    """
    This function generates Poisson binary spike trains.

    Args:
        pars (dict): dictionary with model parameters
        dt (float): time step
        nt (int): number of time steps
        rate (float): rate of firing

    Returns:
       poisson_train: ndarray of generated Poisson binary spike trains (N x number of times steps)
    """

    # generate uniformly distributed random variables
    u_rand = np.random.rand(N,nt)

    # generate Poisson train
    poisson_train = np.int8(1.*(u_rand < rate*dt/1000.0))

    return poisson_train


dt=0.1
tmax=400
nt=int(tmax/dt)+1
rate=50
t = np.linspace(0.0, tmax, nt)
S_in=poisson_generator(1, dt, nt, rate).reshape(t.shape)

plt.plot(t, S_in)
plt.show()
plt.savefig()



