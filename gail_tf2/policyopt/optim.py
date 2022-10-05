import tensorflow as tf
import scipy.sparse.linalg as ssl
from collections import namedtuple
from tensorflow.keras import Model
import numpy as np
from gail_tf2.policyopt import util, nn
from tensorflow.keras.backend import floatx

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
    # m = g.dot(dx)
    m = tf.tensordot(g, dx, 1)
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
    fullstep = step / tf.math.sqrt(.5 * tf.tensordot(step, damped_hvp_func(step), 1) / max_kl + 1e-8)

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
    return tuple(tf.gather(a, subsamp_inds) for a in feed)


NGStepInfo = namedtuple('NGStepInfo', 'obj0, kl0, obj1, kl1, gnorm, bt')
def make_ngstep_func(model, compute_obj_kl, compute_obj_kl_with_grad, compute_kl_hvp):
    '''
    Makes a wrapper for ngstep for classes that implement nn.Model
    Subsamples inputs for fast Hessian-vector products
    '''
    assert isinstance(model, nn.NeuralNet)

    def wrapper(feed, max_kl, damping, subsample_hvp_frac=.1, grad_stop_tol=1e-6, max_cg_iter=10, enable_bt=True):
        assert isinstance(feed, tuple)

        params0 = model.get_params()
        
        # feed = (obsfeat, a, adist, standardized(adv))
        obj0, kl0, objgrad0 = compute_obj_kl_with_grad(feed)
        gnorm = util.maxnorm(objgrad0)
        assert np.allclose(kl0, 0), 'Initial KL divergence is %.7f, but should be 0' % (kl0,)
        # Terminate early if gradient is too small
        if gnorm < grad_stop_tol:
            return NGStepInfo(obj0, kl0, obj0, kl0, gnorm, 0)

        # Data subsampling for Hessian-vector products
        subsamp_feed = feed if subsample_hvp_frac is None else subsample_feed(feed, subsample_hvp_frac)
        def hvpx0_func(v):
            with model.try_params(params0):
                return compute_kl_hvp(subsamp_feed, tf.cast(v, dtype=floatx()))
        # Objective for line search
        def obj_and_kl_func(p):
            with model.try_params(p):
                obj, kl = compute_obj_kl(feed)
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
        obj1, kl1 = compute_obj_kl(feed)
        return NGStepInfo(obj0, kl0, obj1, kl1, gnorm, num_bt_steps)

    return wrapper