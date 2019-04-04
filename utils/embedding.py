import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from config.lstm import EMBEDDING_FILE, EMBEDDING_SIZE


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def get_embeddings():
    """
        Gets all the embeddings from the embedding file
    """
    embeddings = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
    return embeddings


class Embedding:
    def __init__(self, X_train, X_test, max_features, maxlen):
        self.embed_size = EMBEDDING_SIZE  # how big is each word vector
        self.embeddings = get_embeddings()
        self.X_train = X_train
        self.X_test = X_test
        self.max_features = max_features
        self.maxlen = maxlen

        # get preprocessed train and test set
        self.t_X_train, self.t_X_test, self.tokenizer = self.pre_process()

    def pre_process(self):
        """
            Handles preprocessing of data by:
                1. tokenizing the words
                2. Padding the sentences acc to maxlen of words
        :return: tokenized train {X_t} and test sentences {X_te}
        """

        tokenizer = Tokenizer(num_words=self.max_features)

        # Fit tokenizer on train set
        tokenizer.fit_on_texts(list(self.X_train))

        # Tokenize all train and test sentences
        list_tokenized_train = tokenizer.texts_to_sequences(self.X_train)
        list_tokenized_test = tokenizer.texts_to_sequences(self.X_test)

        # Pad sentences
        X_t = pad_sequences(list_tokenized_train, maxlen=self.maxlen)
        X_te = pad_sequences(list_tokenized_test, maxlen=self.maxlen)

        return X_t, X_te, tokenizer

    def generate_embedding_matrix(self):
        """
            Creates embeddings matrix from embeddings
            :return embedding_matrix with dim [number of words X embedding size]
        :return:
        """
        # get all the embeddings
        all_embs = np.stack(list(self.embeddings.values()))

        # mean and standard deviation of the embedding values
        emb_mean, emb_std = all_embs.mean(), all_embs.std()

        # Find the number of words the the train set
        word_index = self.tokenizer.word_index
        rows = min(self.max_features, len(word_index))

        # create embeddings for each word by sampling from a normal distribtion
        # with parameters calculated above.
        embedding_matrix = np.random.normal(emb_mean, emb_std, (rows + 1, self.embed_size))

        for word, i in word_index.items():
            # TODO: replace this with stop words and frequency instead
            if i >= self.max_features:
                continue

            embedding_vector = self.embeddings.get(word)

            # if word is found in the embeddings used
            if embedding_vector is not None:
                # replace row of embeddings matrix with trained embeddings vector
                embedding_matrix[i] = embedding_vector

        return embedding_matrix
