import tensorflow as tf
from tensorflow.keras.backend import floatx
import numpy as np

def flatcat(arrays):
    return tf.concat([tf.reshape(a, [-1]) for a in arrays], axis=0)


def flatgrad(loss_fn, loss_fn_in, vars):
    with tf.GradientTape(persistent=True) as t:
        loss = loss_fn(loss_fn_in)
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