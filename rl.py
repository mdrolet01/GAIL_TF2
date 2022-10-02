import random
import time
import gym
import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, Input
from nn import *
from abc import abstractmethod
from collections import namedtuple
import scipy.sparse.linalg as ssl
import pdb


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


def flatcat(arrays):
    return tf.concat([a.flatten() for a in arrays])


def flatgrad(loss_fn, loss_fn_in, vars):
    with tf.GradientTape(persistent=True) as t:
        loss = loss_fn(loss_fn_in)
    grads = t.gradient(loss, vars, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

def gf(loss_fn, inputs, vars):
    with tf.GradientTape(persistent=True) as t:
        loss = loss_fn(inputs)
    grads = t.gradient(loss, vars, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

def gaussian_kl(means1_N_D, stdevs1_N_D, means2_N_D, stdevs2_N_D):
    D = tf.cast(means1_N_D.shape[1], floatx())
    return (
          .5*(tf.math.reduce_sum(tf.math.square(stdevs1_N_D/stdevs2_N_D), axis=1) +
              tf.math.reduce_sum(tf.math.square((means2_N_D-means1_N_D)/stdevs2_N_D), axis=1) +
          2.*(tf.math.reduce_sum(tf.math.log(stdevs2_N_D), axis=1) - tf.reduce_sum(tf.math.log(stdevs1_N_D), axis=1)) - D
        ))


def gaussian_log_density(means_N_D, stdevs_N_D, x_N_D):
    '''Log density of a Gaussian distribution with diagonal covariance (specified as standard deviations).'''
    D = tf.cast(means_N_D.shape[1], floatx())
    lognormconsts_B = -.5*tf.math.reduce_sum(D*tf.math.log(2.*np.pi) + 2.*tf.math.log(stdevs_N_D), axis=1)
    inner_term = tf.math.square((x_N_D - means_N_D) / stdevs_N_D)
    logprobs_B = -.5*tf.math.reduce_sum(inner_term) + lognormconsts_B
    return logprobs_B

def btlinesearch(f, x0, fx0, g, dx, accept_ratio, shrink_factor, max_steps, verbose=False):
    '''
    Find a step size t such that f(x0 + t*dx) is within a factor
    accept_ratio of the linearized function value improvement.

    Args:
        f: the function
        x0: starting point for search
        fx0: the value f(x0). Will be computed if set to None.
        g: search direction, typically the gradient of f at x0
        dx: the largest possible step to take
        accept_ratio: termination criterion
        shrink_factor: how much to decrease the step every iteration
    '''
    if fx0 is None: fx0 = f(x0)
    t = 1.
    m = g.dot(dx)
    if accept_ratio != 0 and m > 0: print('WARNING: %.10f not <= 0' % m)
    num_steps = 0
    while num_steps < max_steps:
        true_imp = f(x0 + t*dx) - fx0
        lin_imp = t*m
        if verbose: true_imp, lin_imp, accept_ratio
        if true_imp <= accept_ratio * lin_imp:
            break
        t *= shrink_factor
        num_steps += 1
    return x0 + t*dx, num_steps


def ngstep(x0, obj0, objgrad0, obj_and_kl_func, hvpx0_func, max_kl, damping, max_cg_iter, enable_bt):
    '''
    Natural gradient step using hessian-vector products

    Args:
        x0: current point
        obj0: objective value at x0
        objgrad0: grad of objective value at x0
        obj_and_kl_func: function mapping a point x to the objective and kl values
        hvpx0_func: function mapping a vector v to the KL Hessian-vector product H(x0)v
        max_kl: max kl divergence limit. Triggers a line search.
        damping: multiple of I to mix with Hessians for Hessian-vector products
        max_cg_iter: max conjugate gradient iterations for solving for natural gradient step
    '''
    assert x0.ndim == 1 and x0.shape == objgrad0.shape

    # Solve for step direction
    damped_hvp_func = lambda v: hvpx0_func(v) + damping*v
    hvpop = ssl.LinearOperator(shape=(x0.shape[0], x0.shape[0]), matvec=damped_hvp_func)
    step, _ = ssl.cg(hvpop, -objgrad0, maxiter=max_cg_iter)
    fullstep = step / np.sqrt(.5 * step.dot(damped_hvp_func(step)) / max_kl + 1e-8)

    # Line search on objective with a hard KL wall
    if not enable_bt:
        return x0+fullstep, 0

    def barrierobj(p):
        obj, kl = obj_and_kl_func(p)
        return np.inf if kl > 2*max_kl else obj
    xnew, num_bt_steps = btlinesearch(
        f=barrierobj,
        x0=x0,
        fx0=obj0,
        g=objgrad0,
        dx=fullstep,
        accept_ratio=.1, shrink_factor=.5, max_steps=10)
    return xnew, num_bt_steps



def subsample_feed(feed, frac):
    assert isinstance(feed, tuple) and len(feed) >= 1
    assert isinstance(frac, float) and 0. < frac <= 1.
    l = feed[0].shape[0]
    assert all(a.shape[0] == l for a in feed), 'All feed entries must have the same length'
    subsamp_inds = np.random.choice(l, size=int(frac*l))
    return tuple(a[subsamp_inds,...] for a in feed)


NGStepInfo = namedtuple('NGStepInfo', 'obj0, kl0, obj1, kl1, gnorm, bt')
def make_ngstep_func(model, compute_obj_kl, compute_obj_kl_with_grad, compute_kl_hvp):
    '''
    Makes a wrapper for ngstep for classes that implement nn.Model
    Subsamples inputs for fast Hessian-vector products
    '''
    assert isinstance(model, Model)

    def wrapper(feed, max_kl, damping, subsample_hvp_frac=.1, grad_stop_tol=1e-6, max_cg_iter=10, enable_bt=True):
        assert isinstance(feed, tuple)

        params0 = model.get_params()
        obj0, kl0, objgrad0 = compute_obj_kl_with_grad(*feed)
        gnorm = maxnorm(objgrad0)
        assert np.allclose(kl0, 0), 'Initial KL divergence is %.7f, but should be 0' % (kl0,)
        # Terminate early if gradient is too small
        if gnorm < grad_stop_tol:
            return NGStepInfo(obj0, kl0, obj0, kl0, gnorm, 0)

        # Data subsampling for Hessian-vector products
        subsamp_feed = feed if subsample_hvp_frac is None else subsample_feed(feed, subsample_hvp_frac)
        def hvpx0_func(v):
            with model.try_params(params0):
                hvp_args = subsamp_feed + (v,)
                return compute_kl_hvp(*hvp_args)
        # Objective for line search
        def obj_and_kl_func(p):
            with model.try_params(p):
                obj, kl = compute_obj_kl(*feed)
            return -obj, kl
        params1, num_bt_steps = ngstep(
            x0=params0,
            obj0=-obj0,
            objgrad0=-objgrad0,
            obj_and_kl_func=obj_and_kl_func,
            hvpx0_func=hvpx0_func,
            max_kl=max_kl,
            damping=damping,
            max_cg_iter=max_cg_iter,
            enable_bt=enable_bt)
        model.set_params(params1)
        obj1, kl1 = compute_obj_kl(*feed)
        return NGStepInfo(obj0, kl0, obj1, kl1, gnorm, num_bt_steps)

    return wrapper


class Policy(Model):
    def __init__(self, obsfeat_space, action_space, num_actiondist_params, enable_obsnorm, varscope_name):
        super(Policy, self).__init__()
        self.obsfeat_space, self.action_space, self._num_actiondist_params = obsfeat_space, action_space, num_actiondist_params

        with tf.name_scope(varscope_name) as self.__varscope:
            # Action distribution for this current policy
            obsfeat_B_Df = Input(shape=obsfeat_space.shape[0], dtype=floatx())
            self.obsnorm = Standardizer(self.obsfeat_space.shape[0])
            normalized_obsfeat_B_Df = self.obsnorm.standardize_expr(obsfeat_B_Df)
            actiondist_B_Pa = self._make_actiondist_ops(normalized_obsfeat_B_Df)
            self._compute_actiondist_params = Model(inputs=obsfeat_B_Df, outputs=actiondist_B_Pa)
        
        # Only code above this line (i.e. _make_actiondist_ops) is allowed to make trainable variables.
        
        param_vars = self.trainable_variables
        # Reinforcement learning
        input_actions_B_Da = Input(shape=action_space.shape[0], dtype=tf.keras.backend.floatx())
        logprobs_B = self._make_actiondist_logprob_ops(actiondist_B_Pa, input_actions_B_Da)

        # Proposal distribution from old policy
        proposal_actiondist_B_Pa = Input(shape=self.action_space.shape[0])
        proposal_logprobs_B = self._make_actiondist_logprob_ops(proposal_actiondist_B_Pa, input_actions_B_Da)
        
        # Local RL objective
        advantage_B = Input(shape=1, dtype=floatx())
        impweight_B = tf.exp(logprobs_B - proposal_logprobs_B)
        obj = tf.reduce_mean((impweight_B*advantage_B))
        all_inputs = [obsfeat_B_Df, input_actions_B_Da, proposal_actiondist_B_Pa, advantage_B]
        obj_fn = Model(inputs=all_inputs, outputs=obj)

        # test_inputs = [np.array([[1,2]]), np.array([[3]]), np.array([[1]]), np.array([[9]])]
        # kl_inputs = [obsfeat_B_Df, proposal_actiondist_B_Pa]
        # test_inputs2 = [np.array([[1,2]]), np.array([[3]])]

        # KL divergence from old policy
        kl_B = self._make_actiondist_kl_ops(proposal_actiondist_B_Pa, actiondist_B_Pa)
        kl = tf.reduce_mean(kl_B)
        kl_fn = Model(inputs=all_inputs, outputs=kl)
        # kl_fn2 = Model(inputs=kl_inputs, outputs=kl)

        compute_obj_kl = tf.function(lambda _in : (obj_fn(_in), kl_fn(_in)))
        compute_obj_kl_with_grad = tf.function(lambda _in : (obj_fn(_in), kl_fn(_in), flatgrad(obj_fn, _in, param_vars)))
        
        # KL Hessian-vector product
        klgrad_P = tf.function(lambda _in : flatgrad(kl_fn, _in, param_vars))
        # klgrad_P2 = tf.function(lambda _in : flatgrad(kl_fn2, _in, param_vars))
        compute_hvp = tf.function(lambda _in, v_P : flatgrad((lambda __in : tf.reduce_sum(klgrad_P(__in)*v_P)), _in, param_vars))
        self._ngstep = make_ngstep_func(self, compute_obj_kl, compute_obj_kl_with_grad, compute_hvp)

    
    def update_obsnorm(self, obs_B_Do):
        '''Update observation normalization using a moving average'''
        self.obsnorm.update(obs_B_Do)
    
    @tf.function
    def sample_actions(self, obsfeat_B_Df, deterministic=False):
        '''Samples actions conditioned on states'''
        actiondist_B_Pa = self._compute_actiondist_params(obsfeat_B_Df)
        return self._sample_from_actiondist(actiondist_B_Pa, deterministic), actiondist_B_Pa

    # To be overridden
    @abstractmethod
    def _compute_actiondist_params_ops(self, *args): pass
    @abstractmethod
    def _make_actiondist_ops(self, obsfeat_B_Df): pass
    @abstractmethod
    def _make_actiondist_logprob_ops(self, actiondist_B_Pa, input_actions_B_Da): pass
    @abstractmethod
    def _make_actiondist_kl_ops(self, proposal_actiondist_B_Pa, actiondist_B_Pa): pass
    @abstractmethod
    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic): pass
    @abstractmethod
    def _compute_actiondist_entropy(self, actiondist_B_Pa): pass


class GaussianPolicy(Policy):
    def __init__(self, cfg, obsfeat_space, action_space, varscope_name):
        Policy.__init__(
            self,
            obsfeat_space=obsfeat_space,
            action_space=action_space,
            num_actiondist_params=action_space.shape[0]*2,
            enable_obsnorm=True,
            varscope_name=varscope_name)

    def _extract_actiondist_params(self, actiondist_B_Pa):
        means_B_Da = actiondist_B_Pa[:, :self.action_space.shape[0]]
        stdevs_B_Da = actiondist_B_Pa[:, self.action_space.shape[0]:]
        return means_B_Da, stdevs_B_Da

    def _make_actiondist_logprob_ops(self, actiondist_B_Pa, input_actions_B_Da):
        means_B_Da, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        return gaussian_log_density(means_B_Da, stdevs_B_Da, input_actions_B_Da)

    def _make_actiondist_kl_ops(self, proposal_actiondist_B_Pa, actiondist_B_Pa):
        proposal_means_B_Da, proposal_stdevs_B_Da = self._extract_actiondist_params(proposal_actiondist_B_Pa)
        means_B_Da, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        return gaussian_kl(proposal_means_B_Da, proposal_stdevs_B_Da, means_B_Da, stdevs_B_Da)
    
    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic):
        adim = self.action_space.shape[0]
        means_B_Da, stdevs_B_Da = actiondist_B_Pa[:,:adim], actiondist_B_Pa[:,adim:]
        if deterministic:
            return means_B_Da
        stdnormal_B_Da = np.random.randn(actiondist_B_Pa.shape[0], adim)
        assert stdnormal_B_Da.shape == means_B_Da.shape == stdevs_B_Da.shape
        return (stdnormal_B_Da*stdevs_B_Da) + means_B_Da
    
    def _compute_actiondist_params_ops(self, actiondist_B_Pa):
        means_B_Da, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        return tf.function(lambda x : tf.concat([means_B_Da(x), stdevs_B_Da], axis=1))
    
    def _make_actiondist_ops(self, obsfeat_B_Df):
        act_dim = self.action_space.shape[0]
        dense1 = Dense(100, activation='tanh')(obsfeat_B_Df)
        dense2 = Dense(100, activation='tanh')(dense1)
        means_B_Da = Dense(act_dim)(dense2)

        logstdevs_1_Da = tf.Variable(np.full((1, act_dim), -0.5), dtype=floatx())
        stdevs_1_Da = tf.exp(logstdevs_1_Da)
        stdevs_B_Da = tf.ones(shape=(act_dim,))*stdevs_1_Da
        
        return tf.concat([means_B_Da, stdevs_B_Da], axis=1)

class ValueFunc(Model):
    def __init__(self, hidden_spec, obsfeat_space, enable_obsnorm, enable_vnorm, varscope_name, max_kl, damping, time_scale):
        super(ValueFunc, self).__init__()
        self.hidden_spec = hidden_spec
        self.obsfeat_space = obsfeat_space
        self.enable_obsnorm = enable_obsnorm
        self.enable_vnorm = enable_vnorm
        self.max_kl = max_kl
        self.damping = damping
        self.time_scale = time_scale

        with tf.name_scope(varscope_name) as self.__varscope:
            # Action distribution for this current policy
            obsfeat_B_Df = Input(shape=obsfeat_space.shape[0], dtype=floatx())
            self.obsnorm = Standardizer(self.obsfeat_space.shape[0])
            self.vnorm = Standardizer(1)
            t_B = Input(batch_input_shape=(None,1), dtype=floatx())
            scaled_t_B = t_B * self.time_scale
            net_input = tf.concat([obsfeat_B_Df, scaled_t_B], axis=1)
            val_B = self._make_dense_net(net_input)
            self._evaluate_raw = Model(inputs=[obsfeat_B_Df, t_B], outputs=val_B)
        
        param_vars = self.trainable_variables
        target_val_B = Input(shape=1, dtype=floatx())
        old_val_B = Input(shape=1, dtype=floatx())
        all_inputs = [obsfeat_B_Df, t_B, target_val_B, old_val_B]
        obj = -tf.reduce_mean(tf.math.square(val_B - target_val_B))
        obj_fn = Model(inputs=all_inputs, outputs=obj)
        objgrad_P = tf.function(lambda _in : flatgrad(obj_fn, _in, param_vars))
        test_inputs = [np.array([[1,2]], dtype=floatx()), np.array([[3,1]], dtype=floatx()), np.array([[1]], dtype=floatx()), np.array([[1]], dtype=floatx())]
        
        # KL divergence (as Gaussian) and its gradient
        kl = tf.reduce_mean(tf.math.square(old_val_B - val_B))
        kl_fn = Model(inputs=all_inputs, outputs=[kl, obj])
        
        test_tens = [obsfeat_B_Df, t_B, target_val_B, old_val_B]
        objgrad_P = gf(kl_fn, test_tens, param_vars)
        compute_obj_kl = tf.function(lambda _in : (obj_fn(_in), kl_fn(_in)))
        compute_obj_kl_with_grad = tf.function(lambda _in : (obj_fn(_in), kl_fn(_in), flatgrad(obj_fn, _in, param_vars)))

        # KL Hessian-vector product
        klgrad_P = tf.function(lambda _in : flatgrad(kl_fn, _in, param_vars))
        compute_hvp = tf.function(lambda _in, x_P : flatgrad((lambda __in : tf.reduce_sum(klgrad_P(__in)*x_P)), _in, param_vars))
        # self._ngstep = optim.make_ngstep_func(self, compute_obj_kl, compute_obj_kl_with_grad, compute_kl_hvp)
    
    def evaluate(self, obs_B_Do, t_B):
        # ignores the time
        assert obs_B_Do.shape[0] == t_B.shape[0]
        stds = self.obsnorm.standardize(obs_B_Do)
        v1 = self._evaluate_raw([stds, np.array(t_B, dtype=floatx()).reshape(-1,1)])
        return self.vnorm.unstandardize(v1)[:,0]

    def _make_dense_net(self, net_input):
        dense1 = Dense(100, activation='tanh')(net_input)
        dense2 = Dense(100, activation='tanh')(dense1)
        out_layer = Dense(1)(dense2)
        return out_layer

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
        assert all(len(a) == l for a,l in safezip(self.arrays, self.lengths))
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
        time = RaggedArray([np.arange(len(t), dtype=float) for t in trajs])
        return cls(trajs, obs, obsfeat, adist, a, r, time)

    def with_replaced_reward(self, new_r):
        new_trajs = [Trajectory(traj.obs_T_Do, traj.obsfeat_T_Df, traj.adist_T_Pa, traj.a_T_Da, traj_new_r) for traj, traj_new_r in safezip(self.trajs, new_r)]
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


def compute_qvals(r, gamma):
    assert isinstance(r, RaggedArray)
    trajlengths = r.lengths
    # Zero-fill the rewards on the right, then compute Q values
    rewards_B_T = r.padded(fill=0.)
    qvals_zfilled_B_T = discount(rewards_B_T, gamma)
    assert qvals_zfilled_B_T.shape == (len(trajlengths), trajlengths.max())
    return RaggedArray([qvals_zfilled_B_T[i,:l] for i, l in enumerate(trajlengths)]), rewards_B_T


def compute_advantage(r, obsfeat, time, value_func, gamma, lam):
    assert isinstance(r, RaggedArray) and isinstance(obsfeat, RaggedArray) and isinstance(time, RaggedArray)
    trajlengths = r.lengths
    assert np.array_equal(obsfeat.lengths, trajlengths) and np.array_equal(time.lengths, trajlengths)
    B, maxT = len(trajlengths), trajlengths.max()

    # Compute Q values
    q, rewards_B_T = compute_qvals(r, gamma)
    q_B_T = q.padded(fill=np.nan); assert q_B_T.shape == (B, maxT) # q values, padded with nans at the end

    # Time-dependent baseline that cheats on the current batch
    simplev_B_T = np.tile(np.nanmean(q_B_T, axis=0, keepdims=True), (B, 1)); assert simplev_B_T.shape == (B, maxT)
    simplev = RaggedArray([simplev_B_T[i,:l] for i, l in enumerate(trajlengths)])

    # State-dependent baseline (value function)
    v_stacked = value_func.evaluate(obsfeat.stacked, time.stacked); assert len(v_stacked.shape) == 1
    v = RaggedArray(v_stacked, lengths=trajlengths)

    # Compare squared loss of value function to that of the time-dependent value function
    constfunc_prediction_loss = np.var(q.stacked)
    simplev_prediction_loss = np.var(q.stacked-simplev.stacked) #((q.stacked-simplev.stacked)**2).mean()
    simplev_r2 = 1. - simplev_prediction_loss/(constfunc_prediction_loss + 1e-8)
    vfunc_prediction_loss = np.var(q.stacked-v_stacked) #((q.stacked-v_stacked)**2).mean()
    vfunc_r2 = 1. - vfunc_prediction_loss/(constfunc_prediction_loss + 1e-8)

    # Compute advantage -- GAE(gamma, lam) estimator
    v_B_T = v.padded(fill=0.)
    # append 0 to the right
    v_B_Tp1 = np.concatenate([v_B_T, np.zeros((B,1))], axis=1); assert v_B_Tp1.shape == (B, maxT+1)
    delta_B_T = rewards_B_T + gamma*v_B_Tp1[:,1:] - v_B_Tp1[:,:-1]
    adv_B_T = discount(delta_B_T, gamma*lam); assert adv_B_T.shape == (B, maxT)
    adv = RaggedArray([adv_B_T[i,:l] for i, l in enumerate(trajlengths)])
    assert np.allclose(adv.padded(fill=0), adv_B_T)

    return adv, q, vfunc_r2, simplev_r2


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='MountainCarContinuous-v0')
parser.add_argument('--hid', type=int, default=64)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--cpu', type=int, default=1)
parser.add_argument('--steps', type=int, default=4000)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--exp_name', type=str, default='trpo')
args = parser.parse_args()

lam=0.97
max_ep_len=1000
gamma=0.99
time_scale=1./max_ep_len
ac_kwargs=dict(hidden_sizes=[args.hid]*args.l)
env = gym.make(args.env)

max_kl = .01
damping = .1

gp = GaussianPolicy(None, env.observation_space, env.action_space, 'GaussianPolicy')
vf = ValueFunc(None, env.observation_space, True, True, 'ValueFunc', max_kl, damping, time_scale)

# action_space = env.action_space
# act_dim = action_space.shape[0]
# act_limit = action_space.high[0]

# # Action limit for clamping: assumes all dimensions share the same bound!
# act_limit = env.action_space.high[0]

# # Share information about action space with policy architecture
# ac_kwargs = ac_kwargs or {}
# ac_kwargs['action_space'] = env.action_space


o, ep_ret, ep_len = env.reset(), 0, 0
# # act = tf.function(actor)
# @tf.function
# def traceme():
#     return gp.sample_actions(np.array(o, dtype=floatx()))

# # Log the function graph
# log_dir = 'logs'
# writer = tf.summary.create_file_writer(log_dir)
# tf.summary.trace_on(graph=True, profiler=True)
# traceme()
# with writer.as_default():
#     tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)




from contextlib import contextmanager
@contextmanager
def options(options):
  old_opts = tf.config.optimizer.get_experimental_options()
  tf.config.optimizer.set_experimental_options(options)
  try:
    yield
  finally:
    tf.config.optimizer.set_experimental_options(old_opts)

# import timeit
# with options({'constant_folding': True}):
#     print(tf.config.optimizer.get_experimental_options())
#     print("GP time:", timeit.timeit(lambda: gp.sample_actions(tf.expand_dims(o, 0)), number=10000))

def sim_single(policy_fn, obsfeat_fn, max_ep_len):
    obs, obsfeat, actions, actiondists, rewards = [], [], [], [], []
    ob = env.reset()

    for _ in range(max_ep_len):
        obs.append(ob[None,...].copy())
        obsfeat.append(obsfeat_fn(obs[-1]))
        a, adist = policy_fn(obsfeat[-1])
        # agent_outs = sess.run(get_action_ops, feed_dict={x_ph: ob.reshape(1,-1)})
        # a, v_t, logp_t, info_t = agent_outs[0][0], agent_outs[1], agent_outs[2], agent_outs[3:]
        ob, r, d, _ = env.step(a[0,:])
        rewards.append(r)
        actions.append(a)
        actiondists.append(adist)
        if d: break
    
    return Trajectory(np.concatenate(obs), np.concatenate(obsfeat), np.concatenate(actiondists), 
                      np.concatenate(actions), np.array(rewards))

def sim_mp(policy_fn=None, obsfeat_fn=None):
    min_total_sa = 5000
    trajs = []
    num_sa = 0
    while True:
        tr = sim_single(policy_fn, obsfeat_fn, max_ep_len)
        trajs.append(tr)
        num_sa += len(tr)
        if num_sa >= min_total_sa:
            break
    return TrajBatch.FromTrajs(trajs)

import timeit
with options({'constant_folding': True}):
    trajbatch = sim_mp(policy_fn=gp.sample_actions, obsfeat_fn= lambda obs : obs)
    advantages, qvals, vfunc_r2, simplev_r2 = compute_advantage(
        trajbatch.r, trajbatch.obsfeat, trajbatch.time,
        vf, gamma, lam)
