import numpy as np
import tensorflow as tf

from mnist_utils import load_mnist, IMAGE_SIZE, N_CLASSES

N_HIDDEN1 = 256
N_HIDDEN2 = 256
BATCH_SIZE = 200
EPOCHS = 5

x = tf.placeholder(tf.float32, [None, IMAGE_SIZE], name='x')
y = tf.placeholder(tf.float32, [None, N_CLASSES], name='y')

with tf.name_scope('hidden1'):
    W = tf.Variable(tf.truncated_normal([IMAGE_SIZE, N_HIDDEN1], stddev=0.01),
                    name='weights')
    b = tf.Variable(tf.zeros([N_HIDDEN1]), 'biases')
    hidden1 = tf.nn.relu(tf.matmul(x, W) + b)

with tf.name_scope('hidden2'):
    W = tf.Variable(tf.truncated_normal([N_HIDDEN1, N_HIDDEN2], stddev=0.01),
                    name='weights')
    b = tf.Variable(tf.zeros([N_HIDDEN2]), name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, W) + b)

with tf.name_scope('softmax'):
    W = tf.Variable(tf.truncated_normal([N_HIDDEN2, N_CLASSES], stddev=0.01),
                    name='weights')
    b = tf.Variable(tf.zeros([N_CLASSES]), name='biases')
    out = tf.nn.softmax(tf.matmul(hidden2, W) + b)

# loss
cross_entropy = tf.reduce_mean(
    - tf.reduce_sum(y * tf.log(tf.clip_by_value(out, 1e-10, 1.)),
                    reduction_indices=[1]))

# training step
training_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# evaluation
predicted_class = tf.argmax(out, 1)
true_class = tf.argmax(y, 1)
correct_prediction = tf.equal(predicted_class, true_class)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

# initialize all variables
init = tf.initialize_all_variables()

mnist = load_mnist()

train_data = mnist.train.data
train_target = mnist.train.target

n_batches = mnist.train.data.shape[0] / BATCH_SIZE

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(EPOCHS):
        print 'Epoch', epoch + 1

        for batch in range(n_batches):
            start_i = batch * BATCH_SIZE
            end_i = start_i + BATCH_SIZE

            feed_dict = {x: train_data[start_i:end_i],
                         y: train_target[start_i:end_i]}
            sess.run(training_step, feed_dict=feed_dict)

            if batch % 20 == 0:
                feed_dict = {x: mnist.validation.data,
                             y: mnist.validation.target}
                val_acc = sess.run(accuracy, feed_dict=feed_dict)
                print 'Batch {0}: validation accuracy {1}'.format(batch, val_acc)

        perm = range(mnist.train.data.shape[0])
        np.random.shuffle(perm)
        train_data = mnist.train.data[perm]
        train_target = mnist.train.target[perm]

    feed_dict = {x: mnist.test.data,
                 y: mnist.test.target}
    print 'Test-set accuracy', sess.run(accuracy, feed_dict=feed_dict)
