import dataset as ds
import tensorflow as tf
import numpy as np

def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

FEATURE_SIZE = 1024
HIDDEN_UNITS_1 = 700
HIDDEN_UNITS_2 = 300
NUM_LABELS = 26

LEARNING_RATE = 1e-3
LAMBDA = 0.0013
BATCH_SIZE = 500
NUM_ITERATIONS = 60
STOP_ACCURACY = 0.99

x = tf.placeholder(tf.float32, [None, FEATURE_SIZE])
y = tf.placeholder(tf.float32, [None, NUM_LABELS])

W_layer1 = weight_variable([FEATURE_SIZE, HIDDEN_UNITS_1])
b_layer1 = bias_variable([HIDDEN_UNITS_1])

layer1 = tf.nn.relu(tf.matmul(x, W_layer1) + b_layer1)

W_layer2 = weight_variable([HIDDEN_UNITS_1, HIDDEN_UNITS_2])
b_layer2 = bias_variable([HIDDEN_UNITS_2])

layer2 = tf.nn.relu(tf.matmul(layer1, W_layer2) + b_layer2)

W_out = weight_variable([HIDDEN_UNITS_2, NUM_LABELS])
b_out = bias_variable([NUM_LABELS])

h = tf.matmul(layer2, W_out) + b_out

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(h, y) + LAMBDA * tf.nn.l2_loss(W_layer1) + LAMBDA * tf.nn.l2_loss(W_layer2) + LAMBDA * tf.nn.l2_loss(W_out))

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    train_x, train_y = ds.scipy_get_all(set_type='train') 
    cv_x, cv_y = ds.scipy_get_all(set_type='cv')
    test_x, test_y = ds.scipy_get_all(set_type='test')

    print('Training NN with lambda {0}, hu1 {1}, hu2 {2}, batch size {3}'.format(LAMBDA, HIDDEN_UNITS_1, HIDDEN_UNITS_2, BATCH_SIZE))

    for i in range(NUM_ITERATIONS): 
        # shuffle the training set
        p = np.random.permutation(range(len(train_x)))
        train_x, train_y = train_x[p], train_y[p]

        for start in range(0, len(train_x), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={ x: train_x[start:end], y: train_y[start:end] })

        current_accuracy = accuracy.eval(session=sess, feed_dict={ x: train_x, y: train_y })
        current_cost = sess.run(cost, feed_dict={x: train_x, y: train_y})

        print("Training accuracy of {0:.2f}% on {1}th iteration with {2} cost".format(current_accuracy * 100, i, current_cost))

        if current_accuracy > STOP_ACCURACY:
            break

        cv_accuracy = accuracy.eval(session=sess, feed_dict={ x: cv_x, y: cv_y })

        print("Cross-validation accuracy of {0:.2f}%".format(cv_accuracy * 100))

        if i % 10 == 0 and i > 0:
            test_accuracy = accuracy.eval(session=sess, feed_dict={ x: test_x, y: test_y })
            print("Test accuracy of {0:.2f}%".format(test_accuracy * 100))

    cv_accuracy = accuracy.eval(session=sess, feed_dict={ x: cv_x, y: cv_y })
    test_accuracy = accuracy.eval(session=sess, feed_dict={ x: test_x, y: test_y })

    print("Cross validation accuracy of {0:.2f}%".format(cv_accuracy*100)) 
    print("Test accuracy of {0:.2f}%".format(test_accuracy * 100))

    saver.save(sess, 'models/twolayers_moreunits')
