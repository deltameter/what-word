from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

FEATURE_SIZE = 784
HIDDEN_UNITS = 128
NUM_LABELS = 10

LEARNING_RATE = .3
BATCH_SIZE = 100
NUM_ITERATIONS = 10000

x = tf.placeholder(tf.float32, [None, FEATURE_SIZE])
y = tf.placeholder(tf.float32, [None, 10])

W_layer1 = weight_variable([FEATURE_SIZE, HIDDEN_UNITS])
b_layer1 = bias_variable([HIDDEN_UNITS])

layer1 = tf.nn.relu(tf.matmul(x, W_layer1) + b_layer1)

W_layer2 = weight_variable([HIDDEN_UNITS, HIDDEN_UNITS])
b_layer2 = bias_variable([HIDDEN_UNITS])

layer2 = tf.nn.relu(tf.matmul(layer1, W_layer2) + b_layer2)

W_out = weight_variable([HIDDEN_UNITS, NUM_LABELS])
b_out = bias_variable([NUM_LABELS])

h = tf.nn.relu(tf.matmul(layer2, W_out) + b_out)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(h, y))

train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(NUM_ITERATIONS):
    train_x, train_y = mnist.train.next_batch(BATCH_SIZE)

    if i % 100 == 0:
        current_accuracy = accuracy.eval(session=sess, feed_dict={ x: train_x, y: train_y })
        print("Accuracy of %s on %s iteration" % (current_accuracy, i))

    sess.run(train_step, feed_dict={ x: train_x, y: train_y })
    

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y:mnist.test.labels}))
