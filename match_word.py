import numpy as np

words = open('dataset/words.txt', 'r').read().splitlines()

def match_word(word_matrix):

    best_error = 100

    NOT_FOUND = ("Word not found", 100)

    def guess_word(word_matrix, current_error, depth):
        nonlocal best_error

        if depth > 6:
            # probably barking up the wrong tree at this point, try other permutations
            return NOT_FOUND 

        # pointless to continue if our current error is already greater than a previously found word
        if current_error > best_error:
            return NOT_FOUND

        # turn the matrix into letters to check if it's an English word
        guess = np.argmax(word_matrix, 1).tolist()
        guess = list(map(lambda x: chr(x + 65), guess))
        guess = ''.join(guess).lower() 

        if guess in words:
            best_error = current_error

            return guess.upper(), current_error

        # if it's not a word, do tree recursive guessing
        # replace each letter by the next likeliest letter
        # then check if that is a word, and if not, repeat

        new_matrices = []

        for i in range(word_matrix.shape[0]):
            new_matrix = np.matrix.copy(word_matrix)
            character = new_matrix[i]
            likeliest_possibility, likeliest_index = np.amax(character), np.argmax(character)
            character[likeliest_index] = 0
            new_possibility = np.amax(character)

            new_matrices.append((new_matrix, likeliest_possibility - new_possibility))
        
        # sorting by least error will let us cut down on computation time
        # because many recursive calls will be culled by current_error checking
        new_matrices.sort(key=lambda x: -x[1])

        return min([guess_word(matrix[0], current_error + matrix[1], depth + 1) for matrix in new_matrices], key=lambda x: x[1])

    return guess_word(word_matrix, 0, 0)[0]
