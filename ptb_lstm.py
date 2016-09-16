import numpy as np
import tensorflow as tf

from ptb_reader import ptb_raw_data, ptb_batches


# training
batch_size = 20  # exactly what it sounds like
learning_rate = 1.0  # initial value for the learning rate
max_epoch = 4  # number of epochs trained with the initial learning rate
training_epochs = 13  # total number of epochs for training
lr_decay = 0.5  # the decay of the learning rate for each epoch after max_epoch
max_grad_norm = 5.0  # the maximum permissible norm of the gradient

# network parameters
lstm_size = 200  # number of LSTM units
n_layers = 2  # number of LSTM layers
n_steps = 20  # number of unrolled steps of LSTM
keep_prob = 1.0  # probability of keeping weights in the dropout layer


#############
# Load data #
#############

train, valid, test, vocab = ptb_raw_data('datasets/ptb/')
x_batches, y_batches = ptb_batches(train, batch_size, n_steps)

# vocabulary size
vocab_size = vocab


#########
# Graph #
#########

words = tf.placeholder(tf.int32, [batch_size, n_steps], name='words_in_input')
targets = tf.placeholder(tf.int32, [batch_size, n_steps], name='target_words')

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=0.0)
lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
    lstm_cell, output_keep_prob=keep_prob)

cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * n_layers)

initial_state = cell.zero_state(batch_size, tf.float32)


with tf.variable_scope('RNN'):
    embedding = tf.get_variable('embedding', [vocab_size, lstm_size])
    embedded_words = tf.nn.embedding_lookup(embedding, words)

    inputs = [tf.squeeze(input_, [1])
              for input_ in tf.split(1, n_steps, embedded_words)]

    outputs, last_state = tf.nn.rnn(cell, inputs, dtype=tf.float32)

    concat_outputs = tf.concat(1, outputs)  # first arg = concat_dim
    output = tf.reshape(concat_outputs, [-1, lstm_size])

    softmax_w = tf.get_variable('softmax_w', [lstm_size, vocab_size], dtype=tf.float32)
    softmax_b = tf.get_variable('softmax_b', [vocab_size], dtype=tf.float32)

    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities = tf.nn.softmax(logits)

########
# Loss #
########

# Weighted cross-entropy loss for a sequence of logits
# Args:
#   logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
#   targets: List of 1D batch-sized int32 Tensors of the same length as logits.
#   weights: List of 1D batch-sized float-Tensors of the same length as logits.

loss = tf.nn.seq2seq.sequence_loss_by_example(
    [logits],
    [tf.reshape(targets, [-1])],
    [tf.ones([batch_size * n_steps], dtype=tf.float32)])

cost = tf.reduce_sum(loss) / (batch_size)

final_state = last_state

##################
# Training steps #
##################

lr = tf.Variable(0.0, trainable=False, name='learning_rate')

trainable_vars = tf.trainable_variables()

gradients, _ = tf.clip_by_global_norm(
    tf.gradients(cost, trainable_vars),  # Constructs symbolic partial derivatives of sum of `ys` w.r.t. x in `xs`.
    max_grad_norm)

optimizer = tf.train.GradientDescentOptimizer(lr)

train_op = optimizer.apply_gradients(zip(gradients, trainable_vars))

# initialize all ops
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    summary_writer = tf.train.SummaryWriter('graph/ptb/run1', sess.graph)

    costs = 0.0
    iters = 0

    state = sess.run(initial_state)
    print state.dtype
    print state.shape
    sess.run(tf.assign(lr, learning_rate))

    for epoch in range(training_epochs):
        if epoch > max_epoch:
            sess.run(
                tf.assign(
                    lr, learning_rate * (lr_decay * (training_epochs - epoch)))
                )

        for i in range(len(x_batches)):
            feed_dict = {
                words: x_batches[i],
                targets: y_batches[i],
                initial_state: state
            }
            training_loss, state, _ = sess.run([cost, final_state, train_op], feed_dict)
            print 'epoch', epoch + 1, 'current bacth', i, 'training_loss:', training_loss

            costs += training_loss
            iters += n_steps
