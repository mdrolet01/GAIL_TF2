
# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Oracles (experts) for car tasks."""

import random
import keras.backend as K
from keras.models import model_from_json
import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


class ParticleOracle(py_policy.PyPolicy):

  def __init__(self,
               env,
               multimodal = False,
               goal_threshold = 0.01):
    super(ParticleOracle, self).__init__(env.time_step_spec(),
                                         env.action_spec())
    self._env = env
    self._np_random_state = np.random.RandomState(0)
    json_file = open('./Actor_model_architecture.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.actor = model_from_json(loaded_model_json)
    self.actor.load_weights("./DDPG_actor_model_750.h5")
    self.reset()

  def reset(self):
    return

  def _action(self, time_step,
              policy_state):

    state = time_step.observation
    act = self.actor.predict(state.reshape((1,2)))[0]

    return policy_step.PolicyStep(action=act)
