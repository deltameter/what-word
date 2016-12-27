import dataset as ds
import tensorflow as tf
import numpy as np

weight_init = tf.random_normal_initializer(0, 0.1)
regularizer = tf.contrib.layers.l2_regularizer(0.8)

def weight_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=weight_init, regularizer=regularizer)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

FEATURE_SIZE = 1024
HIDDEN_UNITS = 512
NUM_LABELS = 26

LEARNING_RATE = 3e-4
LAMBDA = 1e-3
BATCH_SIZE = 2000
NUM_ITERATIONS = 1000
STOP_ACCURACY = 0.97

CHECK_GRAD = False

x = tf.placeholder(tf.float32, [None, FEATURE_SIZE])
y = tf.placeholder(tf.float32, [None, NUM_LABELS])

W_layer1 = weight_variable('W_layer1', [FEATURE_SIZE, HIDDEN_UNITS])
b_layer1 = bias_variable([HIDDEN_UNITS])

layer1 = tf.nn.relu(tf.matmul(x, W_layer1) + b_layer1)

# W_layer2 = weight_variable('W_layer2', [HIDDEN_UNITS, HIDDEN_UNITS])
# b_layer2 = bias_variable([HIDDEN_UNITS])

# layer2 = tf.nn.relu(tf.matmul(layer1, W_layer2) + b_layer2)

W_out = weight_variable('W_out', [HIDDEN_UNITS, NUM_LABELS])
b_out = bias_variable([NUM_LABELS])

h = tf.matmul(layer1, W_out) + b_out

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(h, y))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(h, y)
        # + LAMBDA * (
        # tf.nn.l2_loss(W_layer1) +
        # tf.nn.l2_loss(W_layer2) +
        # tf.nn.l2_loss(W_out)))

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    train_x, train_y = ds.scipy_get_all(BATCH_SIZE, NUM_LABELS) 
    test_x, test_y = ds.scipy_get_all(100, NUM_LABELS, test=True)

    for i in range(NUM_ITERATIONS):

        if i % 10 == 0:
            current_accuracy = accuracy.eval(session=sess, feed_dict={ x: train_x, y: train_y })
            current_cost = sess.run(cost, feed_dict={x: train_x, y: train_y})
            print("accuracy of {0:.2f}% on {1}th iteration with {2} cost".format(current_accuracy * 100, i, current_cost))
            if current_accuracy > STOP_ACCURACY:
                break

        if i % 100 == 0 and i > 0:
            test_accuracy = accuracy.eval(session=sess, feed_dict={ x: test_x, y: test_y })

            print("Test accuracy of {0:.2f}%".format(test_accuracy * 100))

        if not CHECK_GRAD:
            sess.run(train_step, feed_dict={ x: train_x, y: train_y })
        else:
            grad = []
            for _ in range(10):
                sess.run(train_step, feed_dict={ x: train_x, y: train_y })
                grad.append(sess.run(cost, feed_dict={ x: train_x, y: train_y}))
            
            print(grad)
    
    test_accuracy = accuracy.eval(session=sess, feed_dict={ x: test_x, y: test_y })

    print("Test accuracy of {0:.2f}%".format(test_accuracy * 100))
    saver.save(sess, 'models/26chars')
