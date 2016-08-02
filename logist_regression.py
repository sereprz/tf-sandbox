# imports
import tensorflow as tf

from mnist_utils import load_mnist, N_CLASSES

HEIGHT = 28
WIDTH = 28
IMAGE_SIZE = HEIGHT * WIDTH
BATCH_SIZE = 100

# graph
x = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, N_CLASSES], name='y')

with tf.name_scope('logist_regression'):
    W = tf.Variable(initial_value=tf.zeros(shape=[IMAGE_SIZE, N_CLASSES]),
                    name='W')
    b = tf.Variable(initial_value=tf.zeros(shape=[N_CLASSES]), name='b')
    y_hat = tf.nn.softmax(tf.matmul(x, W) + b)

# loss function and evaluation
cross_entropy = tf.reduce_mean(- tf.reduce_sum(y * tf.log(y_hat),
                               reduction_indices=[1]))
predicted_class = tf.argmax(input=y_hat, dimension=1)
true_class = tf.argmax(input=y, dimension=1)
correct_prediction = tf.equal(predicted_class, true_class)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

# train
training_step = tf.train.GradientDescentOptimizer(learning_rate=0.5)\
    .minimize(cross_entropy)