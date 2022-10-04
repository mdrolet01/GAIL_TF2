import argparse

import numpy as np

from environments import rlgymenv
import policyopt
from policyopt import SimConfig, rl, util, nn
import theano

SIMPLE_ARCHITECTURE = '[{"type": "fc", "n": 100}, {"type": "nonlin", "func": "tanh"}, {"type": "fc", "n": 100}, {"type": "nonlin", "func": "tanh"}]'

if __name__ == '__main__':
    import argparse
    import gym
    import time
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


    ac_kwargs=dict(hidden_sizes=[args.hid]*args.l)
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    hidden_sizes=(256, 256)
    activation='elu'

    policy_cfg = rl.GaussianPolicyConfig(
    hidden_spec=SIMPLE_ARCHITECTURE,
    min_stdev=0.,
    init_logstdev=0.,
    enable_obsnorm=True)
    mdp = rlgymenv.RLGymMDP(args.env)

    gp = rl.GaussianPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'NULL')

    action_space = env.action_space
    act_dim = action_space.shape[0]
    act_limit = action_space.high[0]

    # Action limit for clamping: assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs = ac_kwargs or {}
    ac_kwargs['action_space'] = env.action_space


    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    # act = tf.function(actor)
    import timeit
    val = np.expand_dims(o,0)
    print("GP time:", timeit.timeit(lambda: gp.sample_actions(val), number=10000))
    import pdb; pdb.set_trace()