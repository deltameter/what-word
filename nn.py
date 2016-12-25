import dataset as ds
import tensorflow as tf
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

FEATURE_SIZE = 16384
HIDDEN_UNITS = 128
NUM_LABELS = 4

LEARNING_RATE = 3e-5
BATCH_SIZE = 100
NUM_ITERATIONS = 10000

x = tf.placeholder(tf.float32, [None, FEATURE_SIZE])
y = tf.placeholder(tf.float32, [None, NUM_LABELS])

W_layer1 = weight_variable([FEATURE_SIZE, HIDDEN_UNITS])
b_layer1 = bias_variable([HIDDEN_UNITS])

layer1 = tf.nn.relu(tf.matmul(x, W_layer1) + b_layer1)

W_out = weight_variable([HIDDEN_UNITS, NUM_LABELS])
b_out = bias_variable([NUM_LABELS])

h = tf.nn.softmax(tf.matmul(layer1, W_out) + b_out)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(h, y))

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for i in range(NUM_ITERATIONS):
        train_x, train_y = ds.scipy_get(BATCH_SIZE, NUM_LABELS) 

        if i % 10 == 0:
            current_accuracy = accuracy.eval(session=sess, feed_dict={ x: train_x, y: train_y })
            print("accuracy of {0:.2f}% on {1}th iteration".format(current_accuracy * 100, i))

        # sess.run(train_step, feed_dict={ x: train_x, y: train_y })
 
        grad = []
        for _ in range(10):
            sess.run(train_step, feed_dict={ x: train_x, y: train_y })
            grad.append(sess.run(cost, feed_dict={ x: train_x, y: train_y}))
        
        print(grad)
