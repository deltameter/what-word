import numpy as np

words = open('words.txt', 'r').read().splitlines()


def guess_word(word_matrix):
    guess = np.argmax(word_matrix, 1).tolist()
    guess = list(map(lambda x: chr(x + 65), guess))
    guess = ''.join(guess).lower()

    if guess in words:
        print(guess.upper())
    else:
        print('Word not found')

    print(np.argmax(word_matrix, 1).tolist())


def is_word(word_matrix):
    pass
