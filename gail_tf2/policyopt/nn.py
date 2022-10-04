import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.backend import get_value, set_value, floatx
from contextlib import contextmanager
import collections
import h5py
import hashlib
import json
import numpy as np
import os
import os.path
import tables, warnings; warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
from gail_tf2.policyopt import util, tfutil



class NeuralNet(Model):
    def __init__(self):
        super(NeuralNet, self).__init__()

    def set_params(self, x):
        assert len(x.shape) == 1
        pos = 0
        for v in self.trainable_variables:
            val = v.read_value()
            s = tf.size(val)
            v.assign(tf.reshape(x[pos:pos+s], val.shape))
            pos += s
        assert pos == x.shape[0]

    def get_params(self):
        return tfutil.flatcat([v.read_value() for v in self.trainable_variables])
    
    @contextmanager
    def try_params(self, x):
        orig_x = self.get_params()
        self.set_params(x)
        yield
        self.set_params(orig_x)

class Standardizer(Model):
    def __init__(self, dim, eps=1e-6, init_count=0, init_mean=0., init_meansq=1.):
        super(Standardizer, self).__init__()
        self._eps = eps
        self._dim = dim
        self._count = tf.Variable(np.array(float(init_count)), trainable=False, dtype=floatx())
        self._mean_1_D = tf.Variable(np.full((1, self._dim), init_mean), trainable=False, dtype=floatx())
        self._meansq_1_D = tf.Variable(np.full((1, self._dim), init_meansq), trainable=False, dtype=floatx())

    def get_stdev(self):
        stdev_1_D = tf.sqrt(tf.nn.relu(self._meansq_1_D - tf.square(self._mean_1_D)))
        return stdev_1_D[0,:]

    def get_mean(self):
        return get_value(self._mean_1_D)

    def update(self, points_N_D):
        # print('updating')
        assert points_N_D.ndim == 2 and points_N_D.shape[1] == self._dim
        num = points_N_D.shape[0]
        count = float(get_value(self._count))
        a = count/(count+num)
        set_value(self._mean_1_D, a*get_value(self._mean_1_D) + (1.-a)*points_N_D.mean(axis=0, keepdims=True))
        set_value(self._meansq_1_D, a*get_value(self._meansq_1_D) + (1.-a)*(points_N_D**2).mean(axis=0, keepdims=True))
        set_value(self._count, count + num)

    def standardize_expr(self, x_B_D):
        # print('nn standardize_expr called')
        return (x_B_D - self.get_mean()) / (self.get_stdev() + self._eps)

    def unstandardize_expr(self, y_B_D):
        return y_B_D*(self.get_stdev() + self._eps) + self.get_mean()

    def standardize(self, x_B_D):
        assert len(x_B_D.shape) == 2
        return (x_B_D - self.get_mean()) / (self.get_stdev() + self._eps)

    def unstandardize(self, y_B_D):
        assert len(y_B_D.shape) == 2
        return y_B_D*(self.get_stdev() + self._eps) + self.get_mean()

def test_standardizer():
    D = 10
    st = Standardizer(D, eps=0)

    x_N_D = np.random.randn(200, D)
    st.update(x_N_D)

    x2_N_D = np.random.randn(300, D)
    st.update(x2_N_D)

    allx = np.concatenate([x_N_D, x2_N_D], axis=0)
    
    st2 = Standardizer(D, eps=0)
    x2 = 15 + np.random.randn(200, D)
    st2.update(x2)
    sst2 = st2.standardize(x2)
    ust2 = st2.unstandardize(sst2)
    assert np.allclose(ust2, x2)
    
    assert np.allclose(get_value(st._mean_1_D)[0,:], allx.mean(axis=0))
    assert np.allclose(st.get_stdev(), allx.std(axis=0))
    print('ok')

if __name__ == '__main__':
    test_standardizer()