import numpy as np
import nltk

from nltk.tokenize import word_tokenize


class WordEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # dimension of the word2vec library used
        self.dim = word2vec.vector_size

    def fit(self, X, y):
        return self

    def mean_embedding(self, doc):
        return np.mean([self.word2vec[w] for w in word_tokenize(doc) if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)

    def transform(self, X):
        return np.array([
            self.mean_embedding(doc)
            for doc in X
        ])


class LengthVectorier(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            len(doc)
            for doc in X
        ]).reshape(-1, 1)
