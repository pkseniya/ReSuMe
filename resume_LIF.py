from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random

N = 800 # total number of neurons

N=800
neur_ei=np.ones(N)
N_in=int(0.2*N)
inh_inds=np.array(random.sample(range(0,800),N_in))
neur_ei[inh_inds]=0
ex_inds=np.where(neur_ei>0)

De = 5. # std for the stochastic current for excitatory neurons
Di = 2. # std for the stochastic current for inhibitory neurons

# LIF model parameters
p = {}
# membrane capacitance:
p['tau'] = 30 # (ms)
# membrane resistance:
p['R'] = 1 # (MOhm)
# membrane potential threshold:
p['V_t'] = 14   # (mV)
# reversal potential of the leakage current
p['E_L'] = 0  # (mV)
# membrane reset voltage:
p['V_r'] = 13.5   # (mV)
p['t_r_e'] = 3 # (ms)
p['t_r_i'] = 2 # (ms)

# initialize time step and experiment duration:
dt = 1  # time step duration (ms)
tmax = 400 # duration of experiment (ms)
nt = int(tmax/dt)+1 # number of time steps
t = np.linspace(0.0,tmax,nt) # time vector in ms

# injected current (nA)
I0 = 13.5

C_ee=0.3
C_ei=0.2 
C_ie=0.4
C_ii=0.1

def LIF_network(N,dt,tmax,p,W,I0,S_in, neur_ei, D = 18):
    """
    This function implements a simple network of LIF neurons

    Args:
        N (int): total number of neurons
        dt (float): time step (ms)
        tmax (float): final time (ms)
        p (dict): parameters of the LIF model
        W (numpy ndarray): matrix of the synaptic weights
        I0 (float): injected current (pA)
        D (float, optional): coefficient to transform from input signal to
        external current.
        S_in - input signal (sequence of 0 and 1)
    Returns:
        V, spikes: calculated membrane potentials and binary spike trains
    """

    # initialization 
    nt = int(tmax/dt) + 1
    spikes = np.zeros((nt,N)) # binary spike train
    V = np.zeros((nt,N)) # membrane potentials
    
    # parameters of the LIF model
    tau = p['tau']
    
    # initial conditions for membrane potentials
    V[0,:] = p['E_L']
    # refractory time counter
    counter = np.zeros((N, 2))
    counter[:,1]=neur_ei
    
    # time loop
    for it in tqdm(range(nt-1)):
        
        # generate the stochastic external current
        I_ext = I0 + D*S_in[it]
        # calculate the synaptic current 
        I_syn = W @ spikes[it,:].reshape([N,1])
        # get the total current
        IC = I_ext + I_syn
        IC = IC.reshape([N,])
        
        # get all neurons that should be kept contant 
        # during the refraction period
        iref = np.where(counter[:,0]>0)
        idyn = np.where(counter[:,0]==0)
        
        # update the membrane potentials using the Euler scheme
        V[it+1,idyn] = V[it,idyn] + dt/tau*(p['E_L'] - V[it,idyn] + p['R']*IC[idyn])
        
        # refractored membranes are kept at the reset potential value
        V[it+1,iref] = p['V_r']
        counter[iref,0] -= 1
        
        # correct the potentials below the reset value
        ireset = np.where(V[it+1,:] < p['V_r'])
        V[it+1,ireset] = p['V_r']
    
        
        # check all fired neurons on this time step
        ifired = np.where(V[it+1,:] >= p['V_t'])
        if(len(ifired)>0):
          V[it+1,ifired] = p['V_r']
          # update spike train
          spikes[it+1,ifired] += 1.0
          # update refractory counter for all fired neurons
          for g in ifired[0]:
            #print(ifired)
            if(counter[g,1]):
              counter[g,0] += int(p['t_r_e']/dt)
            else: 
              counter[g,0] += int(p['t_r_i']/dt)
          
            #print(counter[g,0])
    print()
    return V, spikes


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

def get_W(C_ee, C_ei, C_ie, C_ii, neur_ei, weight):
  distmat=geom_matrix(N, 10, 10, 8)
  W=np.zeros((N,N))
  prob=np.zeros((N,N))
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
      prob[a,b]=probability(C, dist)

  W = weight*(np.random.random(size=(N,N))>prob)
  return W


def raster_plot(spikes,t):
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
    ax[0].set_xlabel(r'$t$ [ms]', fontsize = 16)
    ax[0].set_ylabel(r'$A$', fontsize = 16)
    

    N = spikes.shape[1]    
    i = 0
    while i < N:
        if spikes[:,i].sum() > 0.:
            t_sp = t[spikes[:,i] > 0.5]  # spike times
            ax[1].plot(t_sp, i * np.ones(len(t_sp)), 'k|', ms=1., markeredgewidth=2)
        i += 1
    plt.xlim([t[0], t[-1]])
    plt.ylim([-0.5, N + 0.5])
    ax[1].set_xlabel(r'$t$ [ms]', fontsize = 16)
    ax[1].set_ylabel(r'# neuron', fontsize = 16)
    
    plt.show()


W = get_W(C_ee, C_ei, C_ie, C_ii, neur_ei, 1)
V, spikes = LIF_network(N,dt,tmax,p,W,I0,S_in, neur_ei, D = 18)
raster_plot(spikes,t)
