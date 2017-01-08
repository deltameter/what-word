import numpy as np

words = open('dataset/words.txt', 'r').read().splitlines()

def match_word(word_matrix):

    MAX_DEPTH = 6
    best_error = 100
    EPSILON = 1e-5
    NOT_FOUND = ("Word not found", 100)

    def replace_char_matrix(original_matrix, index):
        new_matrix = np.matrix.copy(original_matrix)
        character = new_matrix[index]
        likeliest_possibility, likeliest_index = np.amax(character), np.argmax(character)
        character[likeliest_index] = 0
        new_possibility = np.amax(character)

        return (new_matrix, likeliest_possibility - new_possibility)

    def guess_word(word_matrix, current_error, depth, lock_letter=None):
        nonlocal best_error
        
        if depth > MAX_DEPTH:
            # probably barking up the wrong tree at this point, try other permutations
            return NOT_FOUND 

        # pointless to continue if our current error is already greater than a previously found word
        if current_error - EPSILON > best_error:
            return NOT_FOUND

        # turn the matrix into letters to check if it's an English word
        guess = np.argmax(word_matrix, 1).tolist()
        guess = list(map(lambda x: chr(x + 65), guess))
        guess = ''.join(guess).lower() 
        
        if guess in words:
            best_error = current_error

            return guess.upper(), current_error

        # most errors are only off by one, so before tree recursion, try to just swap individual letters
        if depth == 0:
            print(guess)
            for i in range(word_matrix.shape[0]):
                rep_char = replace_char_matrix(word_matrix, i)
                single_guess = guess_word(rep_char[0], rep_char[1], depth + 1, lock_letter=i)
            # don't have to do anything with the result, since the entry in current_error will prune
            # alot of the tree recursive calls
            
        if lock_letter is not None:
            char_change = replace_char_matrix(word_matrix, lock_letter)
            return guess_word(char_change[0], current_error + char_change[1], depth + 1, lock_letter)

        # if it's not a word, do tree recursive guessing
        # replace each letter by the next likeliest letter
        # then check if that is a word, and if not, repeat

        new_matrices = []

        for i in range(word_matrix.shape[0]):
            new_matrices.append(replace_char_matrix(word_matrix, i))
        
        # sorting by least error will let us cut down on computation time
        # because many recursive calls will be culled by current_error checking
        new_matrices.sort(key=lambda x: x[1])
        return min([guess_word(matrix[0], current_error + matrix[1], depth + 1) for matrix in new_matrices], key=lambda x: x[1])

    return guess_word(word_matrix, 0, 0)[0]

def get_error(word, word_matrix):
    assert len(word) == word_matrix.shape[0], 'Word and word matrix need to be same length'
    word = word.upper()

    error = 0
    for i in range(len(word)):
        error += np.amax(word_matrix[i]) - word_matrix[i][ord(word[i]) - 65]

    return error

def match_word_tractable(word_matrix):
    # turn the matrix into letters to check if it's an English word
    guess = np.argmax(word_matrix, 1).tolist()
    guess = list(map(lambda x: chr(x + 65), guess))
    guess = ''.join(guess)

    if guess.lower() in words:
        return guess

    length = word_matrix.shape[0]
    candidate_words = list(filter(lambda x: len(x) == length, words))

    best_guess = ("", 100)

    for word in candidate_words:
        error = get_error(word, word_matrix)
        if error < best_guess[1]:
            best_guess = (word, error)

    return best_guess[0].upper()
