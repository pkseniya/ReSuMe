# -*- coding: utf-8 -*-
"""ReSuMe_LIF_tests.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MNy-GV4znW61corvkmMSgR5TFCOmTvbm
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

def generator(N, dt, nt, rate):
    """
    This function generates S_in and S_d
    Args:
        pars (dict): dictionary with model parameters
        dt (float): time step
        nt (int): number of time steps
        rate (float): rate of firing

    Returns:
       train: ndarray of generated Poisson binary spike trains (N x number of times steps)
    """

    # generate uniformly distributed random variables
    u_rand = np.random.rand(N,nt)

    # generate Poisson train
    train = np.int8(1.*(u_rand < rate*dt/1000.0))

    return train

dt=1
tmax=400
nt=int(tmax/dt)+1
rate=50
t = np.linspace(0.0, tmax, nt)
S_in=generator(1, dt, nt, rate).reshape(t.shape)

plt.plot(t, S_in)

"""LIF network"""

N = 800 # total number of neurons

neur_ei=np.ones(N) #array of excitatory/inhibitory neuron flags (neur_ei[i]=1 - neuron i is excitatory, neur_ei[i]=0 - neuron i is inhibitory)
N_in=int(0.2*N)
inh_inds=np.array(random.sample(range(0,800),N_in))
neur_ei[inh_inds]=0
ex_inds=np.where(neur_ei>0)

neur_ei=np.append(neur_ei, 1) #we append the input artificial neuron that "gave us S_in"

# LIF model parameters
p = {}
# membrane capacitance:
p['tau'] = 30. # (ms)
# membrane resistance:
p['R'] = 1 # (GOhm)
# membrane potential threshold:
p['V_t'] = 20.   # (mV)
# reversal potential of the leakage current
p['E_L'] = 13.5  # (mV)
# membrane reset voltage:
p['V_r'] = 13.5   # (mV)
p['t_r_e'] = 3. # (ms)
p['t_r_i'] = 2. # (ms)

# injected current (nA)
I0 = 13.5

C_ee=0.3
C_ei=0.2
C_ie=0.4
C_ii=0.1

def LIF_network(N,dt,tmax,p,W,I0,S_in, neur_ei, D_ex, D_in):
    """
    This function implements a simple network of LIF neurons

    Args:
        N (int): total number of neurons
        dt (float): time step (ms)
        tmax (float): final time (ms)
        p (dict): parameters of the LIF model
        W (numpy ndarray): matrix of the synaptic weights
        I0 (float): injected current
        D_ex, D_in (float): amplitudes of stochastic current for exc. and inh. neurons
        S_in - input signal (sequence of 0 and 1)
    Returns:
        V, spikes: calculated membrane potentials and binary spike trains
    """

    # initialization
    neur_ei_local=neur_ei[0:-1]
    nt = int(tmax/dt) + 1
    spikes = np.zeros((nt,N+1)) # binary spike train
    V = np.zeros((nt,N)) # membrane potentials

    D = np.zeros((N+1,1))
    D[neur_ei==1] = D_ex
    D[neur_ei==0] = D_in
    D[-1]=0

    # parameters of the LIF model
    tau = p['tau']

    # initial conditions
    V[0,:] = p['E_L']
    spikes[:,-1]=S_in
    # refractory time counter
    counter = np.zeros((N, 2))
    counter[:,1]=neur_ei_local

    # time loop
    for it in tqdm(range(nt-1)):

        # generate the external current
        I_ext = I0 + D*np.random.normal(size=(N+1,1))
        # calculate the synaptic current
        I_syn = W @ spikes[it,:].reshape([N+1,1])
        # get the total current
        IC_big = I_ext + I_syn
        IC_big = IC_big.reshape([N+1,])
        IC = IC_big[0:-1]

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
        if(len(ifired[0])>0):
          #print('fire',V[it+1,ifired], it+1)
          V[it+1,ifired] = p['V_r']
          # update spike train
          spikes[it+1,ifired] += 1.0
          # update refractory counter for all fired neurons
          for g in ifired[0]:

            if(counter[g,1]):
              counter[g,0] += int(p['t_r_e']/dt)
            else:
              counter[g,0] += int(p['t_r_i']/dt)


    print()
    return V, spikes

def D(a,b):
  """
  Distance between 2 points a and b in 3D space
  a, b - np.array([x,y,z])
  """
  r = a-b
  return np.sqrt(np.sum(r**2))

def probability(C, dist, lmb=2):
  """
  Probability of connection (according to NMC model)
  """
  power = -(dist**2)/(lmb**2)
  return C*np.exp(power)

def geom_matrix(N, size_x, size_y, size_z):
  """
  Generation of the matrix of N uniformly distributed points in 3D space (Box: size_x X size_y X size_z)
  """
  coords = np.zeros((N,3))
  z=0
  for k in range(N):
    coords[k,0]=k%size_x
    coords[k,1]=(k//size_y)%size_y
    coords[k,2]=z
    if(coords[k,1]==size_y-1)and(coords[k,0]==size_x-1):
      z+=1
  return coords

def get_W(N, C_ee, C_ei, C_ie, C_ii, neur_ei, weight):
  """
  Generation of weight matrix
  """
  distmat=geom_matrix(N, 10, 10, 8)
  W=np.zeros((N+1,N+1))
  prob=np.zeros((N+1,N+1))
  input_coord=np.array([5,5,-1])
  distmat = np.append(distmat, input_coord)
  for a in range(N+1):
    for b in range(N+1):
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


  W = weight*(np.random.random(size=(N+1,N+1))<prob)

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

Weight = 0.5
W = get_W(N, C_ee, C_ei, C_ie, C_ii, neur_ei, Weight)

Stochastic_current_amplitude_ex=5.
Stochastic_current_amplitude_in=3.
V, spikes = LIF_network(N,dt,tmax,p,W,I0,S_in, neur_ei, Stochastic_current_amplitude_ex, Stochastic_current_amplitude_in)

raster_plot(spikes,t)