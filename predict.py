import tensorflow as tf
from dataset import image_to_vector
import numpy as np
import sys

x = tf.placeholder(tf.float32)

W_layer1 = tf.placeholder(tf.float32)
b_layer1 =  tf.placeholder(tf.float32)

layer1 = tf.nn.relu(tf.matmul(x, W_layer1) + b_layer1)

W_out = tf.placeholder(tf.float32)
b_out = tf.placeholder(tf.float32)

h = tf.matmul(layer1, W_out) + b_out

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    restorer = tf.train.import_meta_graph('models/4chars.meta')
    restorer.restore(sess, tf.train.latest_checkpoint('models/'))

    model_vars = tf.trainable_variables()
    for i in range(len(model_vars)):
        model_vars[i] = model_vars[i].eval()

    while True:
        try:
            src = input()
            single = image_to_vector(src)
            single = np.reshape(single, (1, single.size))
            a = sess.run(h, feed_dict={ x: single, W_layer1: model_vars[0], b_layer1: model_vars[1], W_out: model_vars[2], b_out: model_vars[3]})
            print(a)
            print(a[0][0])
            print(chr(np.argmax(a) + 65))
        except EOFError:
            print()
            break
        except FileNotFoundError:
            print('file not found. try again :)')
