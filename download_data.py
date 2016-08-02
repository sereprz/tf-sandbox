import os
import urllib
import tensorflow as tf


def maybe_download(url, filename, dirname):
    '''Download MNIST from url if it's not already there

        :type filename: string
        :param filename: name of the file to be downloaded

        :type dirname: string
        :param dirname: directory for storing the dataset
    '''
    if not tf.gfile.Exists(dirname):
        tf.gfile.MakeDirs(dirname)
    filepath = os.path.join(dirname, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.urlretrieve(url + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.Size()
        print 'Successfully downloaded', filename, size, 'bytes.'

    return filepath
