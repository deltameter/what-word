# Feed the NIST dataset into TensorFlow

import tensorflow as tf
import scipy.misc
import numpy as np
import random

reader = tf.WholeFileReader()

def read_file(filename_queue):
    key, value = reader.read(filename_queue)
    image = tf.image.decode_png(value, channels=1)
    return image

def get_alphabet(batch_size):
    """
    Returns 26 instances in the dataset. One for each character of the alphabet
    """
    #the location of the start of capital chars in ASCII
    OFFSET = 65
    filenames = []
    images = []
    one_hots = []

    for _ in range(batch_size):
        #get a random image
        char = random.randrange(26)
        image_tag = random.randrange(0, 300)
        filenames.append('characters/{:x}/hsf_0_{:0>5d}.png'.format(char + OFFSET, image_tag))
        one_hots.append(char)

    filename_queue = tf.train.string_input_producer(filenames)

    for i in range(batch_size):
        images.append(read_file(filename_queue))
        one_hots[i] = tf.one_hot(one_hots[i], 26)

    return images, one_hots

def scipy_get(batch_size, num_chars):
    OFFSET = 65

    inputs = np.ndarray(shape=(batch_size, 16384), dtype=int)
    labels = []

    for image in range(batch_size):
        char = random.randrange(num_chars)
        image_tag = random.randrange(0, 300)

        inputs[image] = scipy.misc.imread('characters/{:x}/hsf_0_{:0>5d}.png'.format(char + OFFSET, image_tag), flatten=True).flatten()
        labels.append(char)

    inputs = inputs / inputs.max(axis=0)

    outputs = np.eye(num_chars)[labels] 

    return inputs, outputs
