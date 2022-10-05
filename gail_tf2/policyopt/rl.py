from gail_tf2.policyopt import nn, util, tfutil, optim, ContinuousSpace, FiniteSpace, RaggedArray, TrajBatch, Trajectory
from collections import namedtuple
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, Input
from tensorflow.keras.backend import floatx
from gail_tf2.policyopt.nn import NeuralNet, Standardizer
import gym
import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete
import numpy as np
from abc import abstractmethod


class Policy(NeuralNet):
    def __init__(self, obsfeat_space, action_space, num_actiondist_params, enable_obsnorm, varscope_name):
        super(Policy, self).__init__()
        self.obsfeat_space, self.action_space, self._num_actiondist_params = obsfeat_space, action_space, num_actiondist_params

        with tf.name_scope(varscope_name) as self.__varscope:
            # Action distribution for this current policy
            obsfeat_B_Df = Input(shape=obsfeat_space.dim, dtype=floatx())
            with tf.name_scope('obsnorm'):
                if enable_obsnorm:
                    self.obsnorm = Standardizer(self.obsfeat_space.dim)
                else:
                    raise NotImplementedError
            normalized_obsfeat_B_Df = self.obsnorm.standardize_expr(obsfeat_B_Df)
            actiondist_B_Pa = self._make_actiondist_ops(normalized_obsfeat_B_Df)
            self._compute_actiondist_params = Model(inputs=obsfeat_B_Df, outputs=actiondist_B_Pa)
        # Only code above this line (i.e. _make_actiondist_ops) is allowed to make trainable variables.
        param_vars = self.trainable_variables
        # Reinforcement learning
        input_actions_B_Da = Input(shape=action_space.dim, dtype=floatx())
        logprobs_B = self._make_actiondist_logprob_ops(actiondist_B_Pa, input_actions_B_Da)

        # Proposal distribution from old policy
        proposal_actiondist_B_Pa = Input(shape=actiondist_B_Pa.shape[1], dtype=floatx())
        proposal_logprobs_B = self._make_actiondist_logprob_ops(proposal_actiondist_B_Pa, input_actions_B_Da)
        
        # Local RL objective
        advantage_B = Input(shape=1, dtype=floatx())
        impweight_B = tf.exp(logprobs_B - proposal_logprobs_B)
        obj = tf.reduce_mean((impweight_B*advantage_B))
        all_inputs = [obsfeat_B_Df, input_actions_B_Da, proposal_actiondist_B_Pa, advantage_B]
        obj_fn = Model(inputs=all_inputs, outputs=obj)

        # tin = [np.ones((3,2), dtype=floatx()), np.ones((3,1),dtype=floatx()), np.ones((3,2), dtype=floatx()), np.ones((3,1),dtype=floatx())]

        # KL divergence from old policy
        kl_B = self._make_actiondist_kl_ops(proposal_actiondist_B_Pa, actiondist_B_Pa)
        kl = tf.reduce_mean(kl_B)
        kl_fn = Model(inputs=all_inputs, outputs=kl)

        compute_obj_kl = tf.function(lambda _in : (obj_fn(_in), kl_fn(_in)))
        compute_obj_kl_with_grad = tf.function(lambda _in : (obj_fn(_in), kl_fn(_in), tfutil.flatgrad(obj_fn, _in, param_vars)))

        # KL Hessian-vector product
        klgrad_P = tf.function(lambda _in : tfutil.flatgrad(kl_fn, _in, param_vars))

        # tin = [np.array([[1,2]], dtype=floatx()), np.array([[3]], dtype=floatx()), np.array([[1]], dtype=floatx()), np.array([[1]], dtype=floatx())]
        # compute_kl_hvp = tf.function(lambda _in, v_P : tfutil.flatgrad((lambda __in : tf.reduce_sum(klgrad_P(__in)*v_P)), _in, param_vars))
        compute_kl_hvp = tf.function(lambda _in, v_P : tfutil.flatgrad((lambda _v: tf.reduce_sum(klgrad_P(_in)*_v)), v_P, param_vars))
        
        self._ngstep = optim.make_ngstep_func(self, compute_obj_kl, compute_obj_kl_with_grad, compute_kl_hvp)

    
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

GaussianPolicyConfig = namedtuple('GaussianPolicyConfig', 'hidden_spec, min_stdev, init_logstdev, enable_obsnorm')

class GaussianPolicy(Policy):
    def __init__(self, cfg, obsfeat_space, action_space, varscope_name):
        assert isinstance(cfg, GaussianPolicyConfig)
        assert isinstance(obsfeat_space, ContinuousSpace) and isinstance(action_space, ContinuousSpace)
        
        self.cfg = cfg
        Policy.__init__(
            self,
            obsfeat_space=obsfeat_space,
            action_space=action_space,
            num_actiondist_params=action_space.dim*2,
            enable_obsnorm=cfg.enable_obsnorm,
            varscope_name=varscope_name)
        

    def _make_actiondist_ops(self, obsfeat_B_Df):
        dense1 = Dense(64, activation='tanh')(obsfeat_B_Df)
        dense2 = Dense(64, activation='tanh')(dense1)
        means_B_Da = Dense(self.action_space.dim)(dense2)

        self.__logstdevs_1_Da = tf.Variable(np.full((1, self.action_space.dim), self.cfg.init_logstdev), dtype=floatx(), name='logstdevs_1_Da')
        stdevs_1_Da = tf.exp(self.__logstdevs_1_Da)
        stdevs_B_Da = tf.ones_like(means_B_Da)*stdevs_1_Da
        return tf.concat([means_B_Da, stdevs_B_Da], axis=1)

    def _extract_actiondist_params(self, actiondist_B_Pa):
        means_B_Da = actiondist_B_Pa[:, :self.action_space.dim]
        stdevs_B_Da = actiondist_B_Pa[:, self.action_space.dim:]
        return means_B_Da, stdevs_B_Da

    def _make_actiondist_logprob_ops(self, actiondist_B_Pa, input_actions_B_Da):
        means_B_Da, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        return tfutil.gaussian_log_density(means_B_Da, stdevs_B_Da, input_actions_B_Da)

    def _make_actiondist_kl_ops(self, proposal_actiondist_B_Pa, actiondist_B_Pa):
        proposal_means_B_Da, proposal_stdevs_B_Da = self._extract_actiondist_params(proposal_actiondist_B_Pa)
        means_B_Da, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        return tfutil.gaussian_kl(proposal_means_B_Da, proposal_stdevs_B_Da, means_B_Da, stdevs_B_Da)
    
    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic):
        adim = self.action_space.dim
        means_B_Da, stdevs_B_Da = actiondist_B_Pa[:,:adim], actiondist_B_Pa[:,adim:]
        if deterministic:
            return means_B_Da
        stdnormal_B_Da = np.random.randn(actiondist_B_Pa.shape[0], adim)
        assert stdnormal_B_Da.shape == means_B_Da.shape == stdevs_B_Da.shape
        return (stdnormal_B_Da*stdevs_B_Da) + means_B_Da
    
    def _compute_actiondist_params_ops(self, actiondist_B_Pa):
        means_B_Da, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        return tf.function(lambda x : tf.concat([means_B_Da(x), stdevs_B_Da], axis=1))
    
    def _compute_actiondist_entropy(self, actiondist_B_Pa):
        _, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        return util.gaussian_entropy(stdevs_B_Da)

class ValueFunc(NeuralNet):
    def __init__(self, hidden_spec, obsfeat_space, enable_obsnorm, enable_vnorm, varscope_name, max_kl, damping, time_scale):
        super().__init__()
        self.hidden_spec = hidden_spec
        self.obsfeat_space = obsfeat_space
        self.enable_obsnorm = enable_obsnorm
        self.enable_vnorm = enable_vnorm
        self.max_kl = max_kl
        self.damping = damping
        self.time_scale = time_scale

        with tf.name_scope(varscope_name) as self.__varscope:
            # Action distribution for this current policy
            obsfeat_B_Df = Input(shape=self.obsfeat_space.dim, dtype=floatx())
            self.obsnorm = Standardizer(self.obsfeat_space.dim)
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
        
        # KL divergence (as Gaussian) and its gradient
        kl = tf.reduce_mean(tf.math.square(old_val_B - val_B))
        kl_fn = Model(inputs=all_inputs, outputs=kl)
        
        compute_obj_kl = tf.function(lambda _in : (obj_fn(_in), kl_fn(_in)))
        compute_obj_kl_with_grad = tf.function(lambda _in : (obj_fn(_in), kl_fn(_in), tfutil.flatgrad(obj_fn, _in, param_vars)))

        # KL Hessian-vector product
        klgrad_P = tf.function(lambda _in : tfutil.flatgrad(kl_fn, _in, param_vars))
        # tin = [np.array([[1,2]], dtype=floatx()), np.array([[3]], dtype=floatx()), np.array([[1]], dtype=floatx()), np.array([[1]], dtype=floatx())]
        compute_kl_hvp = tf.function(lambda _in, x_P : tfutil.flatgrad((lambda _x: tf.reduce_sum(klgrad_P(_in)*_x)), x_P, param_vars))
        self._ngstep = optim.make_ngstep_func(self, compute_obj_kl, compute_obj_kl_with_grad, compute_kl_hvp)
    
    def evaluate(self, obs_B_Do, t_B):
        # ignores the time
        assert obs_B_Do.shape[0] == t_B.shape[0]
        _eval = self._evaluate_raw([self.obsnorm.standardize(obs_B_Do), np.array(t_B, dtype=floatx()).reshape(-1,1)])
        return self.vnorm.unstandardize(_eval)[:,0]
    
    def fit_vf(self, obs_B_Do, t_B, y_B):
        # ignores the time
        assert obs_B_Do.shape[0] == t_B.shape[0] == y_B.shape[0]

        # Update normalization
        self.obsnorm.update(obs_B_Do)
        self.vnorm.update(y_B[:,None])

        # Take step
        sobs_B_Do = self.obsnorm.standardize(obs_B_Do)
        feed = (sobs_B_Do, t_B[:,None], self.vnorm.standardize(y_B[:,None]), self._evaluate_raw([sobs_B_Do, t_B[:, None]]))
        stepinfo = self._ngstep(feed, max_kl=self.max_kl, damping=self.damping)
        return [
            ('vf_dl', stepinfo.obj1 - stepinfo.obj0, float), # improvement of penalized objective
            ('vf_kl', stepinfo.kl1, float), # kl cost of solution
            ('vf_gnorm', stepinfo.gnorm, float), # gradient norm
            ('vf_bt', stepinfo.bt, int), # number of backtracking steps
        ]

    def update_obsnorm(self, obs_B_Do):
        self.obsnorm.update(obs_B_Do)

    def _make_dense_net(self, net_input):
        dense1 = Dense(64, activation='tanh')(net_input)
        dense2 = Dense(64, activation='tanh')(dense1)
        out_layer = Dense(1)(dense2)
        return out_layer


def TRPO(max_kl, damping, subsample_hvp_frac=.1, grad_stop_tol=1e-6):

    def trpo_step(policy, params0_P, obsfeat, a, adist, adv):
        feed = (obsfeat, a, adist, util.standardized(adv))
        stepinfo = policy._ngstep(feed, max_kl=max_kl, damping=damping, subsample_hvp_frac=subsample_hvp_frac, grad_stop_tol=grad_stop_tol)
        return [
            ('dl', stepinfo.obj1 - stepinfo.obj0, float), # improvement of penalized objective
            ('kl', stepinfo.kl1, float), # kl cost of solution
            ('gnorm', stepinfo.gnorm, float), # gradient norm
            ('bt', stepinfo.bt, int), # number of backtracking steps
        ]

    return trpo_step


def compute_qvals(r, gamma):
    assert isinstance(r, RaggedArray)
    trajlengths = r.lengths
    # Zero-fill the rewards on the right, then compute Q values
    rewards_B_T = r.padded(fill=0.)
    qvals_zfilled_B_T = util.discount(rewards_B_T, gamma)
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
    adv_B_T = util.discount(delta_B_T, gamma*lam); assert adv_B_T.shape == (B, maxT)
    adv = RaggedArray([adv_B_T[i,:l] for i, l in enumerate(trajlengths)])
    assert np.allclose(adv.padded(fill=0), adv_B_T)
    return adv, q, vfunc_r2, simplev_r2


class SamplingPolicyOptimizer(object):
    def __init__(self, mdp, discount, lam, policy, sim_cfg, step_func, value_func, obsfeat_fn):
        self.mdp, self.discount, self.lam, self.policy = mdp, discount, lam, policy
        self.sim_cfg = sim_cfg
        self.step_func = step_func
        self.value_func = value_func
        self.obsfeat_fn = obsfeat_fn

        self.total_num_sa = 0
        self.total_time = 0.
        self.curr_iter = 0

    def step(self):
        with util.Timer() as t_all:

            # Sample trajectories using current policy
            with util.Timer() as t_sample:
                # At the first iter, sample an extra batch to initialize standardization parameters
                if self.curr_iter == 0:
                    trajbatch0 = self.mdp.sim_mp(
                        policy_fn=lambda obsfeat_B_Df: self.policy.sample_actions(obsfeat_B_Df),
                        obsfeat_fn=self.obsfeat_fn,
                        cfg=self.sim_cfg)
                    self.policy.update_obsnorm(trajbatch0.obsfeat.stacked)
                    self.value_func.update_obsnorm(trajbatch0.obsfeat.stacked)

                trajbatch = self.mdp.sim_mp(
                    policy_fn=lambda obsfeat_B_Df: self.policy.sample_actions(obsfeat_B_Df),
                    obsfeat_fn=self.obsfeat_fn,
                    cfg=self.sim_cfg)
                # TODO: normalize rewards

            # Compute baseline / advantages
            with util.Timer() as t_adv:
                advantages, qvals, vfunc_r2, simplev_r2 = compute_advantage(
                    trajbatch.r, trajbatch.obsfeat, trajbatch.time,
                    self.value_func, self.discount, self.lam)

            # Take a step
            with util.Timer() as t_step:
                params0_P = self.policy.get_params()
                extra_print_fields = self.step_func(
                    self.policy, params0_P,
                    trajbatch.obsfeat.stacked, trajbatch.a.stacked, trajbatch.adist.stacked,
                    advantages.stacked)
                self.policy.update_obsnorm(trajbatch.obsfeat.stacked)

            # Fit value function for next iteration
            with util.Timer() as t_vf_fit:
                if self.value_func is not None:
                    extra_print_fields += self.value_func.fit_vf(
                        trajbatch.obsfeat.stacked, trajbatch.time.stacked, qvals.stacked)

        # Log
        self.total_num_sa += sum(len(traj) for traj in trajbatch)
        self.total_time += t_all.dt
        fields = [
            ('iter', self.curr_iter, int),
            ('ret', trajbatch.r.padded(fill=0.).sum(axis=1).mean(), float), # average return for this batch of trajectories
            # ('discret', np.mean([q[0] for q in qvals]), float),
            # ('ravg', trajbatch.r.stacked.mean(), float), # average reward encountered
            ('avglen', int(np.mean([len(traj) for traj in trajbatch])), int), # average traj length
            ('nsa', self.total_num_sa, int), # total number of state-action pairs sampled over the course of training
            ('ent', self.policy._compute_actiondist_entropy(trajbatch.adist.stacked).mean(), float), # entropy of action distributions
            ('vf_r2', vfunc_r2, float),
            ('tdvf_r2', simplev_r2, float),
            ('dx', util.maxnorm(params0_P - self.policy.get_params()), float), # max parameter difference from last iteration
        ] + extra_print_fields + [
            ('tsamp', t_sample.dt, float), # time for sampling
            ('tadv', t_adv.dt + t_vf_fit.dt, float), # time for advantage computation
            ('tstep', t_step.dt, float), # time for step computation
            ('ttotal', self.total_time, float), # total time
        ]
        self.curr_iter += 1
        return fields


if __name__ == '__main__':
    lam=0.97
    max_ep_len=1000
    gamma=0.99
    time_scale=1./max_ep_len
    env = gym.make('MountainCarContinuous-v0')

    max_kl = .01
    damping = .1

    gp = GaussianPolicy(None, env.observation_space, env.action_space, 'GaussianPolicy')
    vf = ValueFunc(None, env.observation_space, True, True, 'ValueFunc', max_kl, damping, time_scale)

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
            ob, r, d, _ = env.step(a[0,:])
            rewards.append(r)
            actions.append(a)
            actiondists.append(adist)
            if d: break
        
        return Trajectory(np.concatenate(obs), np.concatenate(obsfeat), np.concatenate(actiondists), 
                        np.concatenate(actions), np.array(rewards))

    def sim_mp(policy_fn=None, obsfeat_fn=None):
        min_total_sa = 1500
        trajs = []
        num_sa = 0
        while True:
            tr = sim_single(policy_fn, obsfeat_fn, max_ep_len)
            trajs.append(tr)
            num_sa += len(tr)
            if num_sa >= min_total_sa:
                break
        return TrajBatch.FromTrajs(trajs)



    with options({'constant_folding': True}):
        trajbatch = sim_mp(policy_fn=gp.sample_actions, obsfeat_fn= lambda obs : obs)

        advantages, qvals, vfunc_r2, simplev_r2 = compute_advantage(
            trajbatch.r, trajbatch.obsfeat, trajbatch.time,
            vf, gamma, lam)

        step_func = TRPO(max_kl, damping)

        params0_P = gp.get_params()

        extra_print_fields = step_func(
            gp, params0_P,
            trajbatch.obsfeat.stacked, trajbatch.a.stacked, trajbatch.adist.stacked,
            advantages.stacked[:,None])

        gp.update_obsnorm(trajbatch.obsfeat.stacked)

        extra_print_fields += vf.fit_vf(
            trajbatch.obsfeat.stacked, trajbatch.time.stacked, qvals.stacked)