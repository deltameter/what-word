import tensorflow as tf
from dataset import scipy_get_all, word_image_to_matrix
import numpy as np
from PIL import Image
from match_word import match_word_tractable
import argparse

DEFAULT_FILEPATH = "dataset/words/"

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W_layer1 = tf.placeholder(tf.float32)
b_layer1 =  tf.placeholder(tf.float32)

layer1 = tf.nn.relu(tf.matmul(x, W_layer1) + b_layer1)

W_layer2 = tf.placeholder(tf.float32)
b_layer2 =  tf.placeholder(tf.float32)

layer2 = tf.nn.relu(tf.matmul(layer1, W_layer2) + b_layer2)

W_out = tf.placeholder(tf.float32)
b_out = tf.placeholder(tf.float32)

h = tf.nn.softmax(tf.matmul(layer2, W_out) + b_out)

correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--test', help='display accuracy of test set', action='store_true')
    args = parser.parse_args()

    with tf.Session() as sess:
        sess.run(init)

        restorer = tf.train.import_meta_graph('models/twolayers_moreunits.meta')
        restorer.restore(sess, tf.train.latest_checkpoint('models/'))

        model_vars = tf.trainable_variables()

        for i in range(len(model_vars)):
            model_vars[i] = model_vars[i].eval()

        parameters = {
            W_layer1: model_vars[0], 
            b_layer1: model_vars[1], 
            W_layer2: model_vars[2], 
            b_layer2: model_vars[3], 
            W_out: model_vars[4], 
            b_out: model_vars[5]
        }

        if args.test:
            parameters[x], parameters[y] = scipy_get_all(set_type='test')
            model_accuracy = accuracy.eval(feed_dict=parameters)
            print('Model loaded with test accuracy of {0:.2f}%'.format(model_accuracy * 100))
            
            hypo = h.eval(feed_dict=parameters)
            hypo = np.argmax(hypo, axis=1)
            
            error_dict = {}
            for i in range(len(hypo)):
                if hypo[i] != np.argmax(parameters[y][i]):
                    error_dict[hypo[i]] = error_dict.get(hypo[i], 0) + 1
            print(error_dict)
        else:
            print('Model loaded')

        print('Enter the filepath of the word you\'d like to read. Default filepath is {0}, use ./ to search from current directory.'.format(DEFAULT_FILEPATH))

        while True:
            try:
                src = input()

                if src[0] != '.':
                    src = DEFAULT_FILEPATH + src

                letters, word_image = word_image_to_matrix(src)
                
                parameters[x] = letters
                b = sess.run(h, feed_dict=parameters)
                
                # print the guessed word and show the image to the user
                print(match_word_tractable(b)) 
                Image.fromarray(word_image).show()

            except EOFError:
                print()
                break
            except FileNotFoundError:
                print('file not found. try again :)')
