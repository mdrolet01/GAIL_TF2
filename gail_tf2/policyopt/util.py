import numpy as np

def maxnorm(a):
    return np.abs(a).max()

def safezip(*ls):
    assert all(len(l) == len(ls[0]) for l in ls)
    return zip(*ls)

def discount(r_N_T_D, gamma):
    '''
    Computes Q values from rewards.
    q_N_T_D[i,t,:] == r_N_T_D[i,t,:] + gamma*r_N_T_D[i,t+1,:] + gamma^2*r_N_T_D[i,t+2,:] + ...
    '''
    assert r_N_T_D.ndim == 2 or r_N_T_D.ndim == 3
    input_ndim = r_N_T_D.ndim
    if r_N_T_D.ndim == 2: r_N_T_D = r_N_T_D[...,None]

    discfactors_T = np.power(gamma, np.arange(r_N_T_D.shape[1]))
    discounted_N_T_D = r_N_T_D * discfactors_T[None,:,None]
    q_N_T_D = np.cumsum(discounted_N_T_D[:,::-1,:], axis=1)[:,::-1,:] # this is equal to gamma**t * (r_N_T_D[i,t,:] + gamma*r_N_T_D[i,t+1,:] + ...)
    q_N_T_D /= discfactors_T[None,:,None]

    # Sanity check: Q values at last timestep should equal original rewards
    assert np.allclose(q_N_T_D[:,-1,:], r_N_T_D[:,-1,:])

    if input_ndim == 2:
        assert q_N_T_D.shape[-1] == 1
        return q_N_T_D[:,:,0]
    return q_N_T_D

def standardized(a):
    out = np.copy(a)
    out -= np.mean(a)
    out /= np.std(a) + 1e-8
    return out

def gaussian_entropy(stdevs_N_D):
    d = stdevs_N_D.shape[1]
    return .5*d*(1. + np.log(2.*np.pi)) + np.log(stdevs_N_D).sum(axis=1)

import timeit
class Timer(object):
    def __enter__(self):
        self.t_start = timeit.default_timer()
        return self
    def __exit__(self, _1, _2, _3):
        self.t_end = timeit.default_timer()
        self.dt = self.t_end - self.t_start


class Colors(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
def header(s): print(Colors.HEADER + s + Colors.ENDC)
def warn(s): print(Colors.WARNING + s + Colors.ENDC)
def failure(s): print(Colors.FAIL + s + Colors.ENDC)
