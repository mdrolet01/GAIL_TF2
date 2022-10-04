import tensorflow as tf
import numpy as np


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def write_tfr(filename:str="rldata"):
    filename= filename+".tfrecords"
    writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
    count = 0
    for index in range(15):

        rewards = np.random.rand(100,2)
        actions = np.random.rand(100)

        data = {
            'rewards' : _bytes_feature(serialize_array(rewards)),
            'actions' : _bytes_feature(serialize_array(actions)),
        }
        out = tf.train.Example(features=tf.train.Features(feature=data))
        # out = parse_single_image(image=current_image, label=current_label)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count


def _parse_function(example_proto):
    feature_description = {
        'rewards': tf.io.FixedLenFeature([], tf.string),
        'actions': tf.io.FixedLenFeature([], tf.string),
    }
    content = tf.io.parse_single_example(example_proto, feature_description)
    
    return {
        'rewards' : tf.io.parse_tensor(content['rewards'], out_type=tf.double),
        'actions' : tf.io.parse_tensor(content['actions'], out_type=tf.double)
    }

write_tfr()
rd = tf.data.TFRecordDataset(['rldata.tfrecords'])
pard = rd.map(_parse_function)
for sample in pard.take(1):
    import pdb; pdb.set_trace()