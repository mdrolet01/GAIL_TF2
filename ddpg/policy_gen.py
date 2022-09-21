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

evaluate(num_episodes=1, dataset_path="./dataset")
