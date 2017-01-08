# Feed the NIST dataset into TensorFlow
import tensorflow as tf
import scipy.misc
import numpy as np
import random

SCALE_SIZE = 32

SET_SIZES = {
    'train': 2000,
    'cv': 400,
    'test': 350
}

# the "noise" tolerance, any greyscale color value below this registers as "non-empty" when stripping whitespace
TOLERANCE = 100 

reader = tf.WholeFileReader()


def scipy_get_all(set_type='train'):
    OFFSET = 65
    NUM_CHARS = 26
    
    inputs = np.ndarray(shape=(SET_SIZES[set_type] * NUM_CHARS, SCALE_SIZE ** 2), dtype=float)
    labels = []

    for iteration in range(SET_SIZES[set_type]):
        for char in range(NUM_CHARS): 
            index = iteration * NUM_CHARS + char
            labels.append(char)

            if set_type == 'test':
                image_tag = iteration
                inputs[index] = image_to_vector('dataset/{:x}/test/hsf_4_{:0>5d}.png'.format(char + OFFSET, image_tag))
            else:
                if set_type == 'cv':
                    image_tag = SET_SIZES['train'] + iteration
                else:
                    image_tag = iteration

                inputs[index] = image_to_vector('dataset/{0:x}/train/train_{0:x}_{1:0>5d}.png'.format(char + OFFSET, image_tag))

    inputs = inputs / inputs.max()
    outputs = np.eye(NUM_CHARS)[labels] 

    return inputs, outputs

def word_image_to_matrix(location):
    word = strip_whitespace(scipy.misc.imread(location, flatten=True))
    letters = []
    # find individual letters by using whitespace as a delimiter
    whitespace_delimiter = max(1, int(word.shape[1] * 0.01))
    word_delimiter = max(5, int(word.shape[1] * 0.03))
    last_crop = 0
    empty_cols = 0
    filled_cols = 0
    encountered_letter = False

    mins = np.amin(word, axis=0)

    for i in range(mins.size):
        if mins[i] >= TOLERANCE:
            empty_cols += 1
            if empty_cols >= whitespace_delimiter and encountered_letter:
                # make letter!
                letters.append(word[:, last_crop: i])
                last_crop = i
                empty_cols = 0
                encountered_letter = False
        elif mins[i] < TOLERANCE:
            empty_cols = 0
            filled_cols += 1
            if filled_cols >= word_delimiter:
                encountered_letter = True
                filled_cols = 0

    # since whitespace delimits the letter on the left, and we strip whitespace, add last letter
    if encountered_letter:
        letters.append(word[:, last_crop: i])

    # now strip the whitespace from the letters and scale them to match the training set
    letter_matrix = np.ndarray(shape=(len(letters), SCALE_SIZE ** 2), dtype=float)

    for x in range(len(letters)): 
        letters[x] = strip_whitespace(letters[x])
        letters[x] = scipy.misc.imresize(letters[x], (SCALE_SIZE, SCALE_SIZE)) 
        # scipy.misc.imshow(letters[x])
        letter_matrix[x] = letters[x].flatten()

    letter_matrix = letter_matrix / letter_matrix.max()
    
    # return a scaled image of the word to be displayed to the user
    scaled_word = scipy.misc.imresize(word, (int(word.shape[0] / (word.shape[1] / 784)), 784))

    return letter_matrix, scaled_word

def image_to_vector(location):
    image = scipy.misc.imread(location, flatten=True)
    image = strip_whitespace(image)
    image = scipy.misc.imresize(image, (SCALE_SIZE, SCALE_SIZE))
    return image.flatten()

def strip_whitespace(image, square=False):
    # find the 
    non_empty_columns = np.where(image.min(axis=0) < TOLERANCE)[0]
    non_empty_rows = np.where(image.min(axis=1) < TOLERANCE)[0]

    min_rows, max_rows, min_cols, max_cols = min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns)

    if square:
        row_range, col_range = max_rows - min_rows, max_cols - min_cols

        if row_range < col_range:
            diff = (col_range - row_range) // 2
            new_size = (col_range+1, col_range+1)
            cropped_image = np.full(new_size, 255.0)
            cropped_image[diff:(diff + row_range + 1)] = image[min_rows:max_rows + 1, min_cols:max_cols+1]
            
        elif col_range < row_range:
            diff = (row_range - col_range) // 2
            new_size = (row_range+1, row_range+1)
            cropped_image = np.full(new_size, 255.0)
            cropped_image[:, diff:(diff + col_range + 1)] = image[min_rows:max_rows+1, min_cols:max_cols + 1]
        else:
            cropped_image = image[min_rows:max_rows + 1, min_cols:max_cols + 1]
    else:
        crop_box = (min_rows, max_rows, min_cols, max_cols)

        cropped_image = image[crop_box[0]:crop_box[1] + 1, crop_box[2]:crop_box[3] + 1]
    return cropped_image
