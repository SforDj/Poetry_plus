import tensorflow as tf
import numpy as np
from code.data_handler import data_reader
from code.data_handler import fenci


class Poetry:
    def __init__(self, file, batch_size):
        self.poetry_list = self._poetry_reader(file)
        self.poetry_vectors, self.word_to_int, self.int_to_word = self._gen_poetry_vectors()
        self.batch_size = batch_size
        self.chunk_size = len(self.poetry_vectors) // self.batch_size

    def _poetry_reader(self, file):
        return data_reader.poetry_sentence_reader(file)

    def _gen_poetry_vectors(self):
        words = sorted(set(''.join(self.poetry_list)))
        words.append(' ')
        int_to_word = {i: word for i, word in enumerate(words)}
        word_to_int = {word: i for i, word in int_to_word.items()}
        to_int = lambda word: word_to_int.get(word)
        poetry_vectors = [list(map(to_int, poetry)) for poetry in self.poetry_list]
        return poetry_vectors, word_to_int, int_to_word

    def get_batch(self):
        start = 0
        end = self.batch_size
        for _ in range(self.chunk_size):
            batches = self.poetry_vectors[start: end]
            x_batch = np.full([self.batch_size, max(map(len, batches))], self.word_to_int[' '], np.int32)
            for row in range(self.batch_size):
                x_batch[row, :len(batches[row])] = batches[row]
            y_batch = np.copy(x_batch)
            y_batch[:, :-1], y_batch[:, -1] = x_batch[:, 1:], x_batch[:, 0]
            yield x_batch, y_batch
            start += self.batch_size
            end += self.batch_size



