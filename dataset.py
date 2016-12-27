# Feed the NIST dataset into TensorFlow
import tensorflow as tf
import scipy.misc
import numpy as np
from PIL import Image
import random

INPUT_SIZE = 128
SCALE_SIZE = 32

reader = tf.WholeFileReader()

def scipy_get(batch_size, num_chars, test=False):
    OFFSET = 65
    
    inputs = np.ndarray(shape=(batch_size, SCALE_SIZE ** 2), dtype=int)
    labels = []

    for image in range(batch_size):
        char = random.randrange(num_chars)
        if test:
            image_tag = random.randrange(300, 350)
        else:
            image_tag = random.randrange(0, 300)

        inputs[image] = image_to_vector('dataset/{:x}/hsf_0_{:0>5d}.png'.format(char + OFFSET, image_tag))
        labels.append(char)

    inputs = inputs / inputs.max(axis=0)

    outputs = np.eye(num_chars)[labels] 

    return inputs, outputs

def scipy_get_all(batch_size, num_chars, test=False):
    OFFSET = 65
    
    inputs = np.ndarray(shape=(batch_size * num_chars, SCALE_SIZE ** 2), dtype=float)
    labels = []

    for image in range(batch_size):
        for char in range(num_chars):
            if test:
                image_tag = 2000 + image
            else:
                image_tag = image

            index = image * num_chars + char
            inputs[index] = image_to_vector('dataset/{:x}/train/train_{:x}_{:0>5d}.png'.format(char + OFFSET, char + OFFSET, image_tag))
            labels.append(char)


    inputs = inputs / inputs.max()
    outputs = np.eye(num_chars)[labels] 

    return inputs, outputs

def word_image_to_matrix(location):
    word = scipy.misc.imread(location, flatten=True) 
    num_letters = round(word.shape[1] / INPUT_SIZE)
    print(num_letters)
    letters = np.split(word, num_letters, axis=1)
    word_matrix = np.ndarray(shape=(num_letters, SCALE_SIZE ** 2), dtype=float)

    for x in range(len(letters)): 
        letters[x] = strip_whitespace(letters[x])
        letters[x] = scipy.misc.imresize(letters[x], (SCALE_SIZE, SCALE_SIZE)) 
        word_matrix[x] = letters[x].flatten()
    
    word_matrix = word_matrix / word_matrix.max()
    print(np.array_equal(word_matrix[0], np.ceil(word_matrix[0])))
    return word_matrix

def image_to_vector(location):
    image = scipy.misc.imread(location, flatten=True)
    image = strip_whitespace(image)
    image = scipy.misc.imresize(image, (SCALE_SIZE, SCALE_SIZE))
    return image.flatten()

def strip_whitespace(image):
    non_empty_columns = np.where(image.min(axis=0)<255)[0]
    non_empty_rows = np.where(image.min(axis=1)<255)[0]
    crop_box = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

    cropped_image = image[crop_box[0]:crop_box[1]+1, crop_box[2]:crop_box[3]+1]
    return cropped_image
