import gym
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, Input
from gail_tf2.policyopt.nn import Standardizer
import pdb
from tensorflow.keras import backend as K
from tensorflow.keras.backend import floatx

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

def flatcat(arrays):
    return tf.concat([tf.reshape(a, [-1]) for a in arrays], axis=0)

def flatgrad(loss, vars_):
    return flatcat(tf.gradients(loss, vars_))


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


def mlp(hidden_sizes=(64, 32), activation='relu', output_activation=None,
        layer_norm=False):
    """Creates MLP with the specified parameters."""
    model = tf.keras.Sequential()
    model.add(Dense(units=100, input_shape=(None,2), activation=tf.tanh))
    model.add(tf.keras.layers.Dense(units=hidden_sizes[-1], activation=None))
    if layer_norm:
        model.add(tf.keras.layers.LayerNormalization())
    model.add(tf.keras.layers.Activation(output_activation))

    return model

class ContinuousActor(Model):
    """Actor model for continuous action space."""

    def __init__(self, action_space, hidden_sizes=[100,100],
                    activation=tf.tanh, layer_norm=False):
        super(ContinuousActor, self).__init__()
        self._action_dim = action_space.shape

        self._body = mlp(
            hidden_sizes=list(hidden_sizes),
            activation=activation,
            layer_norm=layer_norm
        )

        self._mu = Dense(self._action_dim[0], name='mean')
        log_init = -0.5 * np.ones(shape=(1,) + self._action_dim, dtype=floatx())
        self._log_std = tf.Variable(initial_value=log_init, trainable=True, name='log_std_dev')

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self._body(inputs)
        mu = self._mu(x)
        log_std = tf.clip_by_value(self._log_std, -20, 20)
        return mu, log_std

    @tf.function
    def action(self, observations):
        mu, log_std = self(observations)
        std = tf.exp(log_std)
        return mu + tf.random.normal(tf.shape(input=mu)) * std


class Policy(Model):
    def __init__(self, obsfeat_space, action_space, num_actiondist_params, enable_obsnorm, varscope_name):
        super(Policy, self).__init__()
        self.obsfeat_space, self.action_space, self._num_actiondist_params = obsfeat_space, action_space, num_actiondist_params
        act_dim = self.action_space.shape[0]
        # Action distribution for this current policy
        obsfeat_B_Df = Input(shape=2, dtype=floatx())
        with tf.name_scope(varscope_name) as self.__varscope:
            self.actor = ContinuousActor(action_space)
            self.actor.build((None,2))
            self.actor.compile(run_eagerly=False)
            mu, sig = self.actor(obsfeat_B_Df)

        # Only code above this line (i.e. _make_actiondist_ops) is allowed to make trainable variables.
        param_vars = self.trainable_variables
        
        # Reinforcement learning
        input_actions_B_Da = Input(shape=action_space.shape[0], dtype=floatx())
        mu_prop = Input(shape=1, dtype=floatx())
        sig_prop = Input(shape=1, dtype=floatx())

        # Proposal distribution from old policy
        logprobs_B = gaussian_log_density(mu, sig, input_actions_B_Da)
        proposal_logprobs_B = gaussian_log_density(mu_prop, sig_prop, input_actions_B_Da)
        
        # Local RL objective
        advantage_B = Input(shape=1, dtype=floatx())
        impweight_B = tf.exp(logprobs_B - proposal_logprobs_B)
        obj = tf.reduce_mean((impweight_B*advantage_B))
        objgrad_P = flatgrad(obj, param_vars)
        
        # KL divergence from old policy
        kl_B = gaussian_kl(mu_prop, sig_prop, mu, sig)
        kl = tf.reduce_mean(kl_B)
        
        compute_obj_kl = K.function([obsfeat_B_Df, input_actions_B_Da, mu_prop, sig_prop, advantage_B], [obj, kl])
        compute_obj_kl_with_grad = K.function([obsfeat_B_Df, input_actions_B_Da, mu_prop, sig_prop, advantage_B], [obj, kl, objgrad_P])
        
        v_P = Input(shape=())
        klgrad_P = flatgrad(kl, param_vars)
        hvpexpr = flatgrad(tf.reduce_sum(klgrad_P*v_P), param_vars)
        hvp = K.function(inputs=[obsfeat_B_Df, mu_prop, sig_prop, v_P], outputs=[hvpexpr])
        compute_hvp = lambda _obsfeat_B_Df, _input_actions_B_Da, mu_prop, sig_prop, _advantage_B, _v_P: hvp(_obsfeat_B_Df, mu_prop, sig_prop, _v_P)
        # self._ngstep = make_ngstep_func(self, compute_obj_kl, compute_obj_kl_with_grad, compute_hvp)
    

    def sample_actions(self, obsfeat_B_Df, deterministic=False):
        return self.actor.action(obsfeat_B_Df)

env = gym.make('MountainCarContinuous-v0')

max_kl = .01
damping = .1

gp = Policy(env.observation_space, env.action_space, None, True, 'GaussianPolicy')

# from contextlib import contextmanager
# @contextmanager
# def options(options):
#   old_opts = tf.config.optimizer.get_experimental_options()
#   tf.config.optimizer.set_experimental_options(options)
#   try:
#     yield
#   finally:
#     tf.config.optimizer.set_experimental_options(old_opts)

o, ep_ret, ep_len = env.reset(), 0, 0
import timeit

# actor = ContinuousActor(env.action_space)
# actor.build(input_shape=(None,2))
# actor.compile(run_eagerly=False)

# with options({'constant_folding': True}):
    # print(tf.config.optimizer.get_experimental_options())

gp.compile(run_eagerly=False)
print("GP time:", timeit.timeit(lambda: gp.actor.action(tf.expand_dims(o, 0)), number=10000))
