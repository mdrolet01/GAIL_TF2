from collections import namedtuple
from gail_tf2.policyopt import util
import numpy as np
import multiprocessing
from time import sleep
import binascii
from tensorflow.keras.backend import floatx

class Trajectory(object):
    __slots__ = ('obs_T_Do', 'obsfeat_T_Df', 'adist_T_Pa', 'a_T_Da', 'r_T')
    def __init__(self, obs_T_Do, obsfeat_T_Df, adist_T_Pa, a_T_Da, r_T):
        assert (
            obs_T_Do.ndim == 2 and obsfeat_T_Df.ndim == 2 and adist_T_Pa.ndim == 2 and a_T_Da.ndim == 2 and r_T.ndim == 1 and
            obs_T_Do.shape[0] == obsfeat_T_Df.shape[0] == adist_T_Pa.shape[0] == a_T_Da.shape[0] == r_T.shape[0]
        )
        self.obs_T_Do = obs_T_Do
        self.obsfeat_T_Df = obsfeat_T_Df
        self.adist_T_Pa = adist_T_Pa
        self.a_T_Da = a_T_Da
        self.r_T = r_T

    def __len__(self):
        return self.obs_T_Do.shape[0]

    # Saving/loading discards obsfeat
    def save_h5(self, grp, **kwargs):
        grp.create_dataset('obs_T_Do', data=self.obs_T_Do, **kwargs)
        grp.create_dataset('adist_T_Pa', data=self.adist_T_Pa, **kwargs)
        grp.create_dataset('a_T_Da', data=self.a_T_Da, **kwargs)
        grp.create_dataset('r_T', data=self.r_T, **kwargs)

    @classmethod
    def LoadH5(cls, grp, obsfeat_fn):
        '''
        obsfeat_fn: used to fill in observation features. if None, the raw observations will be copied over.
        '''
        obs_T_Do = grp['obs_T_Do'][...]
        obsfeat_T_Df = obsfeat_fn(obs_T_Do) if obsfeat_fn is not None else obs_T_Do.copy()
        return cls(obs_T_Do, obsfeat_T_Df, grp['adist_T_Pa'][...], grp['a_T_Da'][...], grp['r_T'][...])

def raggedstack(arrays, fill=0., axis=0, raggedaxis=1):
    '''
    Stacks a list of arrays, like np.stack with axis=0.
    Arrays may have different length (along the raggedaxis), and will be padded on the right
    with the given fill value.
    '''
    assert axis == 0 and raggedaxis == 1, 'not implemented'
    arrays = [a[None,...] for a in arrays]
    assert all(a.ndim >= 2 for a in arrays)

    outshape = list(arrays[0].shape)
    outshape[0] = sum(a.shape[0] for a in arrays)
    outshape[1] = max(a.shape[1] for a in arrays) # take max along ragged axes
    outshape = tuple(outshape)

    out = np.full(outshape, fill, dtype=arrays[0].dtype)
    pos = 0
    for a in arrays:
        out[pos:pos+a.shape[0], :a.shape[1], ...] = a
        pos += a.shape[0]
    assert pos == out.shape[0]
    return out


class RaggedArray(object):
    def __init__(self, arrays, lengths=None):
        if lengths is None:
            # Without provided lengths, `arrays` is interpreted as a list of arrays
            # and self.lengths is set to the list of lengths for those arrays
            self.arrays = arrays
            self.stacked = np.concatenate(arrays, axis=0)
            self.lengths = np.array([len(a) for a in arrays])
        else:
            # With provided lengths, `arrays` is interpreted as concatenated data
            # and self.lengths is set to the provided lengths.
            self.arrays = np.split(arrays, np.cumsum(lengths)[:-1])
            self.stacked = arrays
            self.lengths = np.asarray(lengths, dtype=int)
        assert all(len(a) == l for a,l in util.safezip(self.arrays, self.lengths))
        self.boundaries = np.concatenate([[0], np.cumsum(self.lengths)])
        assert self.boundaries[-1] == len(self.stacked)
    def __len__(self):
        return len(self.lengths)
    def __getitem__(self, idx):
        return self.stacked[self.boundaries[idx]:self.boundaries[idx+1], ...]
    def padded(self, fill=0.):
        return raggedstack(self.arrays, fill=fill, axis=0, raggedaxis=1)


class TrajBatch(object):
    def __init__(self, trajs, obs, obsfeat, adist, a, r, time):
        self.trajs, self.obs, self.obsfeat, self.adist, self.a, self.r, self.time = trajs, obs, obsfeat, adist, a, r, time

    @classmethod
    def FromTrajs(cls, trajs):
        assert all(isinstance(traj, Trajectory) for traj in trajs)
        obs = RaggedArray([t.obs_T_Do for t in trajs])
        obsfeat = RaggedArray([t.obsfeat_T_Df for t in trajs])
        adist = RaggedArray([t.adist_T_Pa for t in trajs])
        a = RaggedArray([t.a_T_Da for t in trajs])
        r = RaggedArray([t.r_T for t in trajs])
        time = RaggedArray([np.arange(len(t), dtype=floatx()) for t in trajs])
        return cls(trajs, obs, obsfeat, adist, a, r, time)

    def with_replaced_reward(self, new_r):
        new_trajs = [Trajectory(traj.obs_T_Do, traj.obsfeat_T_Df, traj.adist_T_Pa, traj.a_T_Da, traj_new_r) for traj, traj_new_r in util.safezip(self.trajs, new_r)]
        return TrajBatch(new_trajs, self.obs, self.obsfeat, self.adist, self.a, new_r, self.time)

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return self.trajs[idx]

    def save_h5(self, f, starting_id=0, **kwargs):
        for i, traj in enumerate(self.trajs):
            traj.save_h5(f.require_group('%06d' % (i+starting_id)), **kwargs)

    @classmethod
    def LoadH5(cls, dset, obsfeat_fn):
        return cls.FromTrajs([Trajectory.LoadH5(v, obsfeat_fn) for k, v in dset.iteritems()])


SimConfig = namedtuple('SimConfig', 'min_num_trajs min_total_sa batch_size max_traj_len')

class MDP(object):
    '''General MDP'''

    @property
    def obs_space(self):
        '''Observation space'''
        raise NotImplementedError

    @property
    def action_space(self):
        '''Action space'''
        raise NotImplementedError

    def new_sim(self, init_state=None):
        raise NotImplementedError

    def sim_single(self, policy_fn, obsfeat_fn, max_traj_len, init_state=None):
        '''Simulate a single trajectory'''
        sim = self.new_sim(init_state=init_state)
        obs, obsfeat, actions, actiondists, rewards = [], [], [], [], []
        for _ in range(max_traj_len):
            obs.append(sim.obs[None,...].copy())
            obsfeat.append(obsfeat_fn(obs[-1]))
            a, adist = policy_fn(obsfeat[-1])
            actions.append(a)
            actiondists.append(adist)
            rewards.append(sim.step(a[0,:]))
            if sim.done: break
        obs_T_Do = np.concatenate(obs); assert obs_T_Do.shape == (len(obs), self.obs_space.storage_size)
        obsfeat_T_Df = np.concatenate(obsfeat); assert obsfeat_T_Df.shape[0] == len(obs)
        adist_T_Pa = np.concatenate(actiondists); assert adist_T_Pa.ndim == 2 and adist_T_Pa.shape[0] == len(obs)
        a_T_Da = np.concatenate(actions); assert a_T_Da.shape == (len(obs), self.action_space.storage_size)
        r_T = np.asarray(rewards); assert r_T.shape == (len(obs),)
        return Trajectory(obs_T_Do, obsfeat_T_Df, adist_T_Pa, a_T_Da, r_T)

    def sim_mp(self, policy_fn, obsfeat_fn, cfg, maxtasksperchild=200):
        '''
        Multiprocessed simulation
        Not thread safe! But why would you want this to be thread safe anyway?
        '''
        num_processes = 1 # cfg.batch_size if cfg.batch_size is not None else multiprocessing.cpu_count()//2

        # Bypass multiprocessing if only using one process
        if num_processes == 1:
            trajs = []
            num_sa = 0
            while True:
                t = self.sim_single(policy_fn, obsfeat_fn, cfg.max_traj_len)
                trajs.append(t)
                num_sa += len(t)
                if len(trajs) >= cfg.min_num_trajs and num_sa >= cfg.min_total_sa:
                    break
            return TrajBatch.FromTrajs(trajs)

        global _global_sim_info
        _global_sim_info = (self, policy_fn, obsfeat_fn, cfg.max_traj_len)

        trajs = []
        num_sa = 0

        with set_mkl_threads(1):
            # Thanks John
            pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=maxtasksperchild)
            pending = []
            done = False
            while True:
                if len(pending) < num_processes and not done:
                    pending.append(pool.apply_async(_rollout))
                stillpending = []
                for job in pending:
                    if job.ready():
                        traj = job.get()
                        trajs.append(traj)
                        num_sa += len(traj)
                    else:
                        stillpending.append(job)
                pending = stillpending
                if len(trajs) >= cfg.min_num_trajs and num_sa >= cfg.min_total_sa:
                    done = True
                    if len(pending) == 0:
                        break
                sleep(.001)
            pool.close()

        assert len(trajs) >= cfg.min_num_trajs and sum(len(traj) for traj in trajs) >= cfg.min_total_sa
        return TrajBatch.FromTrajs(trajs)

_global_sim_info = None
def _rollout():
    try:
        import os, random; random.seed(os.urandom(4)); 
        # np.random.seed(int(os.urandom(4).encode('hex'), 16))
        np.random.seed(int(binascii.hexlify(os.urandom(4)), 16))
        global _global_sim_info
        mdp, policy_fn, obsfeat_fn, max_traj_len = _global_sim_info
        return mdp.sim_single(policy_fn, obsfeat_fn, max_traj_len)
    except KeyboardInterrupt:
        pass

# Stuff for temporarily disabling MKL threading during multiprocessing
# http://stackoverflow.com/a/28293128
import ctypes
mkl_rt = None
try:
    mkl_rt = ctypes.CDLL('libmkl_rt.so')
    mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
    mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads
except OSError: # library not found
    util.warn('MKL runtime not found. Will not attempt to disable multithreaded MKL for parallel rollouts.')
from contextlib import contextmanager
@contextmanager
def set_mkl_threads(n):
    if mkl_rt is not None:
        orig = mkl_get_max_threads()
        mkl_set_num_threads(n)
    yield
    if mkl_rt is not None:
        mkl_set_num_threads(orig)
