import linecache
import numpy as np


class WordEmbedding:

    def __init__(self, embfile):

        self.embfile = embfile

        self.dimensions = 0
        self.embeddings_index = dict()
        self.index_words()
        self.vocabulary_size = len(self.embeddings_index)
        self.embedding_matrix = None

    def index_words(self):
        print('Indexing words...', end='', flush=True)
        with open(self.embfile, encoding="utf8") as file:
            for line_no, line in enumerate(file):
                values = line.split()
                word = values[0]
                self.dimensions = len(values[1:])
                self.embeddings_index[word] = line_no + 1
        print('done')

    def create_embedding_matrix(self, tokenizer):
        print('Create embedding matrix...', end='', flush=True)
        self.vocabulary_size = len(tokenizer.word_index) + 1
        self.embedding_matrix = np.zeros(shape=(self.vocabulary_size, self.dimensions))
        for word, index in tokenizer.word_index.items():
            line_no = self.embeddings_index.get(word)
            if line_no is not None:
                line = linecache.getline(self.embfile, line_no)
                try:
                    values = line.split()
                    if word != values[0]:
                        print('Word-Line mismatch!')
                        continue

                    self.embedding_matrix[index] = np.asarray(values[1:], dtype='float32')
                except Exception:
                    continue
        print('done')

    def sequences_to_embeddings(self, sequences):
        seq_embeddings = np.zeros(shape=(sequences.shape[0], sequences.shape[1], self.dimensions))
        for i, sequence in enumerate(sequences):
            for j, word_index in enumerate(sequence):
                seq_embeddings[i, j, :] = self.embedding_matrix[word_index]
        return seq_embeddings
