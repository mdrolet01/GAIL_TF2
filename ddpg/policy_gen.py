import os

from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.metrics import py_metrics
from tf_agents.utils import example_encoding_dataset
import car_oracles


def evaluate(num_episodes,dataset_path=None):
  env_name = 'MountainCarContinuous-v0'
  env = suite_gym.load(env_name)

  policy = car_oracles.ParticleOracle(env)

  metrics = [
      py_metrics.AverageReturnMetric(buffer_size=num_episodes),
      py_metrics.AverageEpisodeLengthMetric(buffer_size=num_episodes),
  ]
  
  observers = metrics[:]

  observers.append(
      example_encoding_dataset.TFRecordObserver(
          dataset_path,
          policy.collect_data_spec,
          py_mode=True,
          compress_image=True))

  driver = py_driver.PyDriver(env, policy, observers, max_episodes=num_episodes)
  time_step = env.reset()
  initial_policy_state = policy.get_initial_state(1)
  driver.run(time_step, initial_policy_state)

  env.close()

# evaluate(num_episodes=10, dataset_path="./dataset")

# res = example_encoding_dataset.parse_encoded_spec_from_file('dataset.spec')
import numpy as np
# for idx in range(10):
dres = example_encoding_dataset.load_tfrecord_dataset([f'2d_oracle_car_{1}.tfrecord'])
actions_all = np.zeros((10,74,1))
obs_all = np.zeros((10,74,2))
rewards_all = np.zeros((10,74))
lengths = np.array([], dtype=int)
count = 0
frame_count = 0
for dr in dres.take(-1):
    action = dr.action.numpy().flatten()[0]
    ob = dr.observation.numpy().flatten().reshape(1,2)
    reward = dr.reward.numpy().flatten()[0]
    if dr.step_type.numpy().flatten()[0] == 0:
        count = 0
        obs = np.zeros((74,2))
        actions = np.zeros((74,1))
        rewards = np.zeros(74)
    else:
        obs[count,:] = ob
        actions[count,:] = action
        rewards[count] = reward
        count += 1
    
    if dr.step_type.numpy().flatten()[0] == 2:
        lengths = np.append(lengths, count)
        actions_all[frame_count] = actions
        obs_all[frame_count] = obs
        rewards_all[frame_count] = rewards
        frame_count += 1

data = {
    'actions' : actions_all,
    'observations' : obs_all,
    'rewards' : rewards_all,
    'lengths' : lengths
}

import pickle
with open('/root/irl_control_container/libraries/imitation/scripts/ddpg_expert1.pkl', 'wb') as fp:
    pickle.dump(data, fp)
