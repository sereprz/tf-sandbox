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


def ptb_iterator(raw_data, batch_size, n_steps):
    '''
        Iterates on the raw PTB data.

        Generates batch_size pointers into the raw PTB data, and allows minibatch
        iteration along those pointers

        Yields:
        Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
        The second element of the tuple is the same data time-shifted to the
        right by one
    '''

    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i: batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // n_steps

    if epoch_size == 0:
        raise ValueError('epoch_size == 0, decrease batch_size or num_steps')

    for i in range(epoch_size):
        x = data[:, i * n_steps: (i + 1) * n_steps]
        y = data[:, i * n_steps + 1: (i + 1) * n_steps + 1]
        yield (x, y)
