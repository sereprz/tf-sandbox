# imports
import numpy as np
import tensorflow as tf

from mnist_utils import load_mnist, N_CLASSES

HEIGHT = 28
WIDTH = 28
IMAGE_SIZE = HEIGHT * WIDTH
BATCH_SIZE = 200
EPOCHS = 5

# graph
x = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, N_CLASSES], name='y')

with tf.name_scope('logistic') as scope:
    W = tf.Variable(initial_value=tf.zeros(shape=[IMAGE_SIZE, N_CLASSES]),
                    name='weights')
    b = tf.Variable(initial_value=tf.zeros(shape=[N_CLASSES]), name='biases')
    y_hat = tf.nn.softmax(tf.matmul(x, W) + b)

# loss function
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(y * tf.log(tf.clip_by_value(y_hat, 1e-10, 1.)),
                   reduction_indices=[1]))

# train
training_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cross_entropy)

# evaluation
predicted_class = tf.argmax(input=y_hat, dimension=1)
true_class = tf.argmax(input=y, dimension=1)
correct_prediction = tf.equal(predicted_class, true_class)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

# initialize all variables
init = tf.initialize_all_variables()

# load dataset
mnist = load_mnist()

train_data = mnist.train.data
train_target = mnist.train.target
val_data = mnist.validation.data
val_target = mnist.validation.target

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
                feed_dict = {x: val_data,
                             y: val_target}
                acc = sess.run(accuracy, feed_dict=feed_dict)
                print 'Step {0}: validation accuracy {1}'.format(batch, acc)

        perm = np.arange(mnist.train.data.shape[0])
        np.random.shuffle(perm)
        train_data, train_target = train_data[perm], train_target[perm]

    test_acc = sess.run(accuracy,
                        feed_dict={x: mnist.test.data, y: mnist.test.target})
    print 'Test accuracy:', test_acc
