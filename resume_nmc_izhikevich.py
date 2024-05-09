# Define the parameters of the LIF neurons
tau = 30 # Membrane time constant (ms)
V_th = 15 # Threshold potential (mV)
V_reset = 13.5 # Reset potential (mV)
V_rest = 0 #resting potential
t_ref_ex = 3 #ms
t_ref_in = 2 #ms
I0 = 2. #nano ampers - background current
R= 1 #mega ohm
F_in = 0.2 #fraction of randomly chosen inhibitory neurons

C_ee=0.3 #EE
C_ei=0.2 
C_ie=0.4
C_ii=0.1

N=800
neur_ei=np.ones(N)
N_in=int(0.2*N)
inh_inds=np.array(random.sample(range(0,800),N_in))
neur_ei[inh_inds]=0
ex_inds=np.where(neur_ei>0)

De = 5. # std for the stochastic current for excitatory neurons
Di = 2. # std for the stochastic current for inhibitory neurons

def D(a,b):
  r = a-b
  return np.sqrt(np.sum(r**2))

def probability(C, dist, lmb=2):
  power = -(dist**2)/(lmb**2)
  return C*np.exp(power)

def geom_matrix(N, size_x, size_y, size_z):
  coords = np.zeros((N,3))
  z=0
  for k in range(N):
    coords[k,0]=k%size_x
    coords[k,1]=(k//size_y)%size_y
    coords[k,2]=z
    if(coords[k,1]==size_y-1)and(coords[k,0]==size_x-1):
      z+=1
  return coords

def generate_W(C_ee, C_ei, C_ie, C_ii, neur_ei):
  distmat=geom_matrix(N, 10, 10, 8)
  W=np.zeros((N,N))
  for a in range(N):
    for b in range(N):
      if(neur_ei[a])and(neur_ei[b]):
        C=C_ee
      elif(neur_ei[a])and(not neur_ei[b]):
        C=C_ei
      elif(not neur_ei[a])and(neur_ei[b]):
        C=C_ie
      else:
        C=C_ii
      dist=D(distmat[a], distmat[b])
      W[a,b]=probability(C, dist)
  return W

W=generate_W(C_ee, C_ei, C_ie, C_ii, neur_ei)

def initialization(N,Ne,Ni,W, inh_inds, ex_inds):
    """
    This function initializes the parameters arrays
    for excitatory and inhibitory neurons described
    by the Izhikevich model, and also the synapses
    connections matrix W

    Args:
        N (int): total number of neurons
        Ne (int): number of excitatory neurons
        Ni (int): number of inhibitory neurons
        weight (float): synaptic weights magnitude for excitatory synapses connetions
        p (float): connection probability

    Returns:
        a, b, c, d, W: numpy arrays with a, b, c, d coefficients for neurons; W - synapses connection matrix
    """

    # random initialization
    re = np.random.uniform(size=((Ne,1)))
    ri = np.random.uniform(size=((Ni,1)))
    print(Ne)
    # excitatory neurons
    ae = np.ones((Ne,1))*0.02
    be = np.ones((Ne,1))*0.2
    ce = -65.*np.ones((Ne,1)) + 15.*re**2
    de = 8.*np.ones((Ne,1)) - 6.*re**2

    # inhibitory neurons
    ai = np.ones((Ni,1))*0.02 + 0.08*ri
    bi = np.ones((Ni,1))*0.25 - 0.05*ri
    ci = -65.*np.ones((Ni,1))
    di = 2.*np.ones((Ni,1))

    # generate coefficients for all neurons
    a = np.ones((N,1))
    a[inh_inds]=ai
    a[ex_inds]=ae

    b = np.ones((N,1))
    b[inh_inds]=bi
    b[ex_inds]=be

    c = np.ones((N,1))
    c[inh_inds]=ci
    c[ex_inds]=ce

    d = np.ones((N,1))
    d[inh_inds]=di
    d[ex_inds]=de
    # generate synapses matrix
   
    W2 = (np.random.random((N,N))>W)

    return a, b, c, d, W2

def simulation_run(N,Ne,Ni,nt,dt,I0,De,Di,a,b,c,d,W):
    """
    This function simulates a network of Izhikevich neuron
    for a given period of time

    Args:
        N (int): total number of neurons
        Ne (int): number of excitatory neurons
        Ni (int): number of inhibitory neurons
        nt (int): number of time steps
        dt (float): time step
        I0 (float): dc current
        De (float): std of the stochastic current for excitatory neurons
        Di (float): std of the stochastic current for inhibitory neurons
        a (numpy array): array of N parameters a of the Izhinkevich model
        b (numpy array): array of N parameters b of the Izhinkevich model
        c (numpy array): array of N parameters c of the Izhinkevich model
        d (numpy array): array of N parameters d of the Izhinkevich model
        W (numpy ndarray): synapses weights matrix

    Returns:
        spikes: binary spikes trains for N neurons
    """

    # initial conditions
    v = -65.*np.ones((N,1))
    u = b*v
    spikes = np.zeros((nt,N),dtype=np.int8)

    # time loop
    for k in range(0,nt-1):

        # generate stochastic currents
        Ie = De*np.random.uniform(low = -1, high = 1, size = (Ne,1))
        Ii = Di*np.random.uniform(low = -1, high = 1, size = (Ni,1))
        IC = np.concatenate((Ie, Ii), axis = 0)
        # add synaptic currents
        IC += I0 + np.matmul(W, spikes[k,:].reshape([N,1]))

        # check firing and update potentials and adaptive currents
        ifired = np.where(v[:,0] >= 30.)
        spikes[k+1,ifired] += 1
        v[ifired] = c[ifired]
        u[ifired] += d[ifired]

        # double step in 0.5 ms in the Euler scheme
        v = v + 0.5*dt*(0.04*v**2 + 5.*v + 140. - u + IC)
        v = v + 0.5*dt*(0.04*v**2 + 5.*v + 140. - u + IC)

        # update adaptive currents
        u = u + a*(b*v - u)

    return spikes

def run(N,Ne,Ni,dt,tmax,I0,De,Di,W):
    """
    This function runs the simulation and plots the results
    
    Args:
        N (int): total number of neurons
        Ne (int): number of excitatory neurons
        Ni (int): number of inhibitory neurons
        dt (float): time step
        tmax (float): max time step
        I0 (float): DC current
        De (float): std of the stochastic current for excitatory neurons 
        Di (float): std of the stochastic current for inhibitory neurons 
        weight (float): synaptic weights for excitatory neurons 
        p (float): probability that the connection is zero (in range [0,1])
    """
    
    a, b, c, d, W = initialization(N,Ne,Ni,W, inh_inds, ex_inds)
    nt = int(tmax/dt)+1
    spikes = simulation_run(N,Ne,Ni,nt,dt,I0,De,Di,a,b,c,d,W)
    t = np.linspace(0.0,tmax,nt)
    raster_plot(spikes,t,W)



def raster_plot(spikes,t,W):
    """
    This function visualize an average activity of 
    the neuron network and its spike train raster plot

    Args:
        spikes (numpy ndarray): binary spike trains for all neurons
        t (numpy array ): time array in ms
    """
    
    fig, ax = plt.subplots(1,2,figsize=(16,4))
    
    A = np.mean(spikes, axis = 1)
    ax[0].plot(t, A, color = 'tab:blue', linewidth = 2.)
    ax[0].set_xlabel(r'$t$ [ms]', fontsize = 20)
    ax[0].set_ylabel(r'$A$', fontsize = 20)
    ax[0].set_ylim([0,1.1])
    

    N = spikes.shape[1]    
    i = 0
    while i < N:
        if spikes[:,i].sum() > 0.:
            t_sp = t[spikes[:,i] > 0.5]  # spike times
            ax[1].plot(t_sp, i * np.ones(len(t_sp)), 'k|', ms=1., markeredgewidth=2)
        i += 1
    ax[1].set_xlim([t[0], t[-1]])
    ax[1].set_ylim([-0.5, N + 0.5])
    ax[1].set_xlabel(r'$t$ [ms]', fontsize = 20)
    ax[1].set_ylabel(r'# neuron', fontsize = 20)
    #ax[1].set_title('$p = $'+str(p)+', w = '+str(w))

    
    plt.show()


run(N,N-N_in,N_in,dt,tmax,I0,De,Di,W)
