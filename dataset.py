# Feed the NIST dataset into TensorFlow
import tensorflow as tf
import scipy.misc
import numpy as np
import random

SCALE_SIZE = 32

# the "noise" tolerance, any greyscale color value below this registers as "non-empty" when stripping whitespace
TOLERANCE = 230 

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
    letters = []
    # find individual letters by using whitespace as a delimiter
    whitespace_delimiter = int(word.shape[1] * 0.01)
    word_delimiter = int(word.shape[1] * 0.03)

    last_crop = 0
    empty_cols = 0
    filled_cols = 0
    encountered_letter = False

    mins = np.amin(word, axis=0)

    for i in range(mins.size):
        if mins[i] == 255 and encountered_letter:
            empty_cols += 1
            if empty_cols == whitespace_delimiter:
                # make letter!
                letters.append(word[:, last_crop: i])
                last_crop = i
                empty_cols = 0
                encountered_letter = False
        elif mins[i] < 245:
            filled_cols += 1
            if filled_cols == word_delimiter:
                encountered_letter = True
                filled_cols = 0

    # now strip the whitespace from the letters and scale them to match the training set
    letter_matrix = np.ndarray(shape=(len(letters), SCALE_SIZE ** 2), dtype=float)

    for x in range(len(letters)): 
        letters[x] = strip_whitespace(letters[x])
        letters[x] = scipy.misc.imresize(letters[x], (SCALE_SIZE, SCALE_SIZE)) 
        letter_matrix[x] = letters[x].flatten()

    letter_matrix = letter_matrix / letter_matrix.max()
    
    # return a scaled image of the word to be displayed to the user
    scaled_word = scipy.misc.imresize(word, (256, int(word.shape[1] / (word.shape[0] / 256))))

    return letter_matrix, scaled_word

def image_to_vector(location):
    image = scipy.misc.imread(location, flatten=True)
    image = strip_whitespace(image)
    image = scipy.misc.imresize(image, (SCALE_SIZE, SCALE_SIZE))
    return image.flatten()

def strip_whitespace(image):
    non_empty_columns = np.where(image.min(axis=0) < TOLERANCE)[0]
    non_empty_rows = np.where(image.min(axis=1) < TOLERANCE)[0]
    crop_box = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

    cropped_image = image[crop_box[0]:crop_box[1]+1, crop_box[2]:crop_box[3]+1]
    return cropped_image
