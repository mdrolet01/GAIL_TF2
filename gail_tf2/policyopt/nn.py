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

def _hash_name2array(name2array):
    '''
    Hashes a list of (name,array) tuples.
    The hash is invariant to permutations of the list.
    '''
    def hash_array(a):
        return '%.10f,%.10f,%d' % (np.mean(a), np.var(a), np.argmax(a))
    vals = '|'.join('%s %s' for n, h in sorted([(name, hash_array(a)) for name, a in name2array]))
    # return hashlib.sha1('|'.join('%s %s' for n, h in sorted([(name, hash_array(a)) for name, a in name2array]))).hexdigest()
    return hashlib.sha1(vals.encode('utf-8')).hexdigest()

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

    def get_num_params(self):
        return sum(tf.size(v.read_value()) for v in self.trainable_variables)

    def print_trainable_variables(self):
        for v in self.trainable_variables:
            util.header('- %s (%d parameters)' % (v.name, tf.size(v.read_value())))
        util.header('Total: %d parameters' % (self.get_num_params(),))

    @contextmanager
    def try_params(self, x):
        orig_x = self.get_params()
        self.set_params(x)
        yield
        self.set_params(orig_x)
    
    def savehash(self):
        return _hash_name2array([(v.name, np.array(v.read_value())) for v in self.variables])

    def save_h5(self, h5file, key, extra_attrs=None):
        with h5py.File(h5file, 'a') as f:
            if key in f:
                util.warn('WARNING: key %s already exists in %s' % (key, h5file))
                dset = f[key]
            else:
                dset = f.create_group(key)

            for v in self.variables:
                dset[v.name] = np.array(v.read_value())

            dset.attrs['hash'] = self.savehash()
            if extra_attrs is not None:
                for k, v in extra_attrs:
                    if k in dset.attrs:
                        util.warn('Warning: attribute %s already exists in %s' % (k, dset.name))
                    dset.attrs[k] = v

    def load_h5(self, h5file, key):
        with h5py.File(h5file, 'r') as f:
            dset = f[key]

            for v in self.variables:
                assert v.name[0] == '/'; vname = v.name[1:]
                print('Reading', vname)
                if vname in dset:
                    v.set_value(dset[vname][...])
                elif vname+':0' in dset:
                    # Tensorflow saves variables with :0 appended to the name,
                    # so try this for backwards compatibility
                    v.set_value(dset[vname+':0'][...])
                else:
                    raise RuntimeError('Variable %s not found in %s' % (vname, dset))

            h = self.savehash()
            assert h == dset.attrs['hash'].decode(), 'Checkpoint hash %s does not match loaded hash %s' % (dset.attrs['hash'].decode(), h)

def _printfields(fields, sep=' | ', width=8, precision=4, print_header=True):
    names, vals, fmts = [], [], []
    for name, val, typeinfo in fields:
        names.append(name)
        if val is None:
            # display Nones as empty entries
            vals.append('')
            fmts.append('{:%ds}' % width)
        else:
            vals.append(val)
            if typeinfo is int:
                fmts.append('{:%dd}' % width)
            elif typeinfo is float:
                fmts.append('{:%d.%df}' % (width, precision))
            else:
                raise NotImplementedError(typeinfo)
    if print_header:
        header = ((('{:^%d}' % width) + sep) * len(names))[:-len(sep)].format(*names)
        print('-'*len(header))
        print(header)
        print('-'*len(header))
    print(sep.join(fmts).format(*vals))

def _type_to_col(t, pos):
    if t is int: return tables.Int32Col(pos=pos)
    if t is float: return tables.Float32Col(pos=pos)
    raise NotImplementedError(t)


class TrainingLog(object):
    '''A training log backed by PyTables. Stores diagnostic numbers over time and model snapshots.'''

    def __init__(self, filename, attrs):
        if filename is None:
            util.warn('Warning: not writing log to any file!')
            self.f = None
        else:
            # if os.path.exists(filename):
            #     raise RuntimeError('Log file %s already exists' % filename)
            self.f = tables.open_file(filename, mode='w')
            for k, v in attrs: self.f.root._v_attrs[k] = v
            self.log_table = None

        self.schema = None # list of col name / types for display

    def close(self):
        if self.f is not None: self.f.close()

    def write(self, kvt, display=True, **kwargs):
        # Write to the log
        if self.f is not None:
            if self.log_table is None:
                desc = {k: _type_to_col(t, pos) for pos, (k, _, t) in enumerate(kvt)}
                self.log_table = self.f.create_table(self.f.root, 'log', desc)

            row = self.log_table.row
            for k,v,_ in kvt: row[k] = v
            row.append()

            self.log_table.flush()

        if display:
            if self.schema is None:
                self.schema = [(k,t) for k,_,t in kvt]
            else:
                # If we are missing columns, fill them in with Nones
                nonefilled_kvt = []
                kvt_dict = {k:(v,t) for k,v,t in kvt}
                for schema_k, schema_t in self.schema:
                    if schema_k in kvt_dict:
                        v, t = kvt_dict[schema_k]
                        nonefilled_kvt.append((schema_k, v, t)) # check t == schema_t too?
                    else:
                        nonefilled_kvt.append((schema_k, None, schema_t))
                kvt = nonefilled_kvt
            _printfields(kvt, **kwargs)

    def write_snapshot(self, model, key_iter):
        if self.f is None: return

        # Save all variables into this group
        snapshot_root = '/snapshots/iter%07d' % key_iter

        for v in model.variables:
            # assert v.name[0] == '/'
            fullpath = f"{snapshot_root}/{v.name}"
            groupname, arrayname = fullpath.rsplit('/', 1)
            self.f.create_array(groupname, arrayname, np.array(v.read_value()), createparents=True)

        # Store the model hash as an attribute
        self.f.get_node(snapshot_root)._v_attrs.hash = model.savehash()

        self.f.flush()


class Standardizer(Model):
    def __init__(self, dim, eps=1e-6, init_count=0, init_mean=0., init_meansq=1.):
        super(Standardizer, self).__init__()
        self._eps = eps
        self._dim = dim
        self._count = tf.Variable(np.array(float(init_count)), trainable=False, dtype=floatx(), name='count')
        self._mean_1_D = tf.Variable(np.full((1, self._dim), init_mean), trainable=False, dtype=floatx(), name='mean_1_D')
        self._meansq_1_D = tf.Variable(np.full((1, self._dim), init_meansq), trainable=False, dtype=floatx(), name='meansq_1_D')

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