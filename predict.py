import tensorflow as tf
from dataset import image_to_vector, word_image_to_matrix
import numpy as np
from PIL import Image
from match_word import match_word

x = tf.placeholder(tf.float32)

W_layer1 = tf.placeholder(tf.float32)
b_layer1 =  tf.placeholder(tf.float32)

layer1 = tf.nn.relu(tf.matmul(x, W_layer1) + b_layer1)

W_out = tf.placeholder(tf.float32)
b_out = tf.placeholder(tf.float32)

h = tf.nn.softmax(tf.matmul(layer1, W_out) + b_out)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    restorer = tf.train.import_meta_graph('models/26chars.meta')
    restorer.restore(sess, tf.train.latest_checkpoint('models/'))

    model_vars = tf.trainable_variables()

    for i in range(len(model_vars)):
        model_vars[i] = model_vars[i].eval()

    print('Model loaded. Enter the filepath of the word you\'d like to read')

    while True:
        try:
            src = input()

            letters, word_image = word_image_to_matrix(src)

            b = sess.run(h, feed_dict={ x: letters, W_layer1: model_vars[0], b_layer1: model_vars[1], W_out: model_vars[2], b_out: model_vars[3]})
            
            # print the guessed word and show the image to the user
            print(match_word(b)) 
            Image.fromarray(word_image).show()

        except EOFError:
            print()
            break
        except FileNotFoundError:
            print('file not found. try again :)')
