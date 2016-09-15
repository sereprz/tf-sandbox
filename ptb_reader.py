import os
import numpy as np
import collections
import tensorflow as tf


def _read_words(filename):
    '''Returns list of tokens'''
    with tf.gfile.Open(filename, 'rb') as f:
        return f.read().replace('\n', '<eos>').split()


def _build_vocab(filename):
    '''Returns word:id dictionary'''
    tokens = _read_words(filename)

    counter = collections.Counter(tokens)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))  # sort in alphabetical order within decreasing frequency

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    '''Lookup words in file'''
    data = _read_words(filename)
    return [word_to_id[word] for word in data]


def ptb_raw_data(path=None):
    '''Read all ptb files and convert strings to integer ids'''
    train_path = os.path.join(path, 'ptb.train.txt')
    valid_path = os.path.join(path, 'ptb.valid.txt')
    test_path = os.path.join(path, 'ptb.test.txt')

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)

    return train_data, valid_data, test_data, vocabulary


def ptb_batches(raw_data, batch_size, n_steps):
    '''
        Generates minibatches
    '''
    num_batches = len(raw_data) // (batch_size * n_steps)

    # cut off last vaues if they don't make a whle batch
    xdata = np.array(raw_data[:num_batches * batch_size * n_steps])

    # shift xdata by one to the left to predict the next word
    ydata = np.roll(xdata, -1)

    # reshape to [num_batches, n_steps]
    xdata = xdata.reshape(-1, n_steps)
    ydata = ydata.reshape(-1, n_steps)

    # list of batches [batch_size, n_steps]
    x_batches = np.split(xdata, num_batches, 0)
    y_batches = np.split(ydata, num_batches, 0)

    return x_batches, y_batches
