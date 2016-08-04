import numpy as np
import tensorflow as tf

from mnist_utils import N_CLASSES, IMAGE_SIZE, HEIGHT, WIDTH, load_mnist

BATCH_SIZE = 100
EPOCHS = 10

# Input variables placeholders
x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE], name='x')
y = tf.placeholder(tf.float32, shape=[None, N_CLASSES], name='y')

# Architecture
# first convolutional layer (32 features)
# max pooling 2x2
# second convolutional layer (64 features)
# max pooling 2x2
# dense layer (1024 units)
# dropout
# softmax layer (10 classes)

CONV_INPUT_SIZE = 5  # size of the patches in input to the convolution
CONV1_OUTPUT_SIZE = 32
CONV2_OUTPUT_SIZE = 64
POOL_SIZE = 2
DENSE_SIZE = 1024
LR = 0.0001

with tf.name_scope('conv1') as scope:
    W = tf.Variable(
        tf.truncated_normal(
            shape=[CONV_INPUT_SIZE, CONV_INPUT_SIZE, 1, CONV1_OUTPUT_SIZE],
            stddev=0.1),
        name='shared_weights')
    b = tf.Variable(tf.constant(0.1, shape=[CONV1_OUTPUT_SIZE]),
                    name='shared_biases')
    x_image = tf.reshape(x, [-1, HEIGHT, WIDTH, 1])
    conv1 = tf.nn.conv2d(x_image, W, strides=[1, 1, 1, 1], padding='SAME')
    h1_conv = tf.nn.relu(conv1 + b)
    h1_pool = tf.nn.max_pool(h1_conv, ksize=[1, POOL_SIZE, POOL_SIZE, 1],
                             strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('conv2') as scope:
    W = tf.Variable(
        tf.truncated_normal(
            shape=[CONV_INPUT_SIZE, CONV_INPUT_SIZE, CONV1_OUTPUT_SIZE, CONV2_OUTPUT_SIZE],
            stddev=0.1),
        name='shared_weights')
    b = tf.Variable(tf.constant(0.1, shape=[CONV2_OUTPUT_SIZE]),
                    name='shared_biases')
    conv2 = tf.nn.conv2d(h1_pool, W, strides=[1, 1, 1, 1], padding='SAME')
    h2_conv = tf.nn.relu(conv2 + b)
    h2_pool = tf.nn.max_pool(h2_conv, ksize=[1, POOL_SIZE, POOL_SIZE, 1],
                             strides=[1, 2, 2, 1], padding='SAME')
    h2_pool_flat = tf.reshape(h2_pool, [-1, 7 * 7 * CONV2_OUTPUT_SIZE])

with tf.name_scope('dense') as scope:
    W = tf.Variable(
        tf.truncated_normal(
            shape=[7 * 7 * CONV2_OUTPUT_SIZE, DENSE_SIZE],
            stddev=0.1),
        name='weights')
    b = tf.Variable(tf.constant(0.1, shape=[DENSE_SIZE]), name='biases')
    h3_dense = tf.nn.relu(tf.matmul(h2_pool_flat, W) + b)
    keep_prob = tf.placeholder(tf.float32)
    h3_dense_dropout = tf.nn.dropout(h3_dense, keep_prob)

with tf.name_scope('sofmax') as scope:
    W = tf.Variable(
        tf.truncated_normal(shape=[DENSE_SIZE, N_CLASSES], stddev=0.1),
        name='weights')
    b = tf.Variable(tf.constant(0.1, shape=[N_CLASSES]), name='biases')
    y_hat = tf.nn.softmax(tf.matmul(h3_dense_dropout, W) + b)


from tensorflow.python.ops.variables import Variable

for k, v in locals().items():
    if type(v) is Variable or type(v) is tf.Tensor:
        print("{0}: {1}".format(k, v))
print '\n\n'

# loss
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(
        y * tf.log(tf.clip_by_value(y_hat, 1e-20, 1.)),
        reduction_indices=[1]))

# training step
training_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)

# evaluation
predicted_class = tf.argmax(y_hat, 1)
true_class = tf.argmax(y, 1)
correct_predicton = tf.equal(predicted_class, true_class)
accuracy = tf.reduce_mean(tf.cast(correct_predicton, dtype=tf.float32))

# initialize ops
init = tf.initialize_all_variables()

# load data
mnist = load_mnist()

train_data = mnist.train.data
train_target = mnist.train.target

n_batches = train_data.shape[0] / BATCH_SIZE
perm = range(train_data.shape[0])

print '\nLearning rate {0}\nbatch size {1}'.format(LR, BATCH_SIZE)

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(EPOCHS):
        print 'Epoch', epoch + 1

        for batch in range(n_batches):
            start_i = batch * BATCH_SIZE
            end_i = start_i + BATCH_SIZE
            feed_dict = {x: train_data[start_i:end_i],
                         y: train_target[start_i:end_i],
                         keep_prob: 0.5}
            sess.run(training_step, feed_dict=feed_dict)

            if batch % 100 == 0:
                feed_dict = {x: train_data[start_i:end_i],
                             y: train_target[start_i:end_i],
                             keep_prob: 1}
                results = sess.run([cross_entropy, accuracy], feed_dict=feed_dict)
                print 'Step {0}: training error {1}, accuracy {2}'.format(batch, results[0], results[1])

        np.random.shuffle(perm)
        train_data = train_data[perm]
        train_target = train_target[perm]

    feed_dict = {x: mnist.test.data, y: mnist.test.target, keep_prob: 1.0}
    print 'Test accuracy', sess.run(accuracy, feed_dict=feed_dict)
