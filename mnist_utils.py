import gzip
import numpy as np
import collections
import tensorflow as tf

from download_data import maybe_download

HEIGHT = 28
WIDTH = 28
IMAGE_SIZE = HEIGHT * WIDTH
N_CLASSES = 10
N_VALIDATION = 5000
DATA_DIR = 'datasets'
MNIST_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'


Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    '''
        Extract the images into a 4D unint8 numpy array [index, y, x, depth]

        :type filename: str
        :param filename: name of the input file
    '''
    print 'Extracting', filename

    with tf.gfile.Open(filename, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            if magic != 2051:
                raise ValueError('Invalid magic number {0} in MNIST image file: {1}'
                                 .format(magic, filename))

            num_images = _read32(bytestream)
            rows = _read32(bytestream)
            cols = _read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols, 1)

        return data


def dense_to_one_hot(labels_dense, n_classes=N_CLASSES):
    n_labels = labels_dense.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename, one_hot=False, n_classes=N_CLASSES):
    '''
        Extract labels into a 1D uint8 numpy array [index]
    '''
    print 'Extracting', filename

    with tf.gfile.Open(filename, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            if magic != 2049:
                raise ValueError('Invalid magic number {0} in MNIST label file: {1}'
                                 .format(magic, filename))

            num_items = _read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)

            if one_hot:
                return dense_to_one_hot(labels, n_classes)

    return labels


def read_data_sets(dirname, one_hot=False, n_classes=N_CLASSES):
    '''
        Download MNIST if necessary, extracts images and labels, prepare for
        analysis.
    '''
    local_file = maybe_download(MNIST_URL, TRAIN_IMAGES, dirname)
    train_images = extract_images(local_file)
    train_images = train_images.reshape(train_images.shape[0],
                                        train_images.shape[1] * train_images.shape[2])
    train_images = np.multiply(train_images, 1. / 255.)

    local_file = maybe_download(MNIST_URL, TRAIN_LABELS, dirname)
    train_labels = extract_labels(local_file, one_hot, n_classes)

    local_file = maybe_download(MNIST_URL, TEST_IMAGES, dirname)
    test_images = extract_images(local_file)
    test_images = test_images.reshape(test_images.shape[0],
                                      test_images.shape[1] * test_images.shape[2])
    test_images = np.multiply(test_images, 1. / 255.)

    local_file = maybe_download(MNIST_URL, TEST_LABELS, dirname)
    test_labels = extract_labels(local_file, one_hot, n_classes)

    test = Dataset(data=test_images, target=test_labels)
    train = Dataset(data=train_images[N_VALIDATION:],
                    target=train_labels[N_VALIDATION:])
    validation = Dataset(data=train_images[:N_VALIDATION],
                         target=train_labels[:N_VALIDATION])

    return Datasets(train=train, validation=validation, test=test)


def load_mnist(dirname=DATA_DIR, one_hot=True):
    return read_data_sets(dirname, one_hot)
