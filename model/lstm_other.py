"""
took help from: https://www.kaggle.com/snlpnkj/bidirectional-lstm-keras,
                https://www.kaggle.com/liliasimeonova/sentiment-analysis-with-bidirectional-lstm

others - 16426
"""

import numpy as np
import pandas as pd
from keras import Input, Model
from keras.layers import Dense, Embedding, LSTM, Bidirectional, GlobalMaxPool1D, Dropout, Concatenate
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from config.lstm import K, LABELS, MISCLASSIFIED_FILE, TRAIN_TEST_SPLIT, EPOCHS, EMBEDDING_FILE, INCLUDE_TYPE_FEATURE, \
    EMBEDDING_SIZE
from model.data_handler import DataHandler
from utils.plot import plot_confusion_multiple
from utils.process_data import get_top_k_indices


def top_k_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=K)


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def get_embeddings():
    """
        Gets all the embeddings from the embedding file
    """
    embeddings = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
    return embeddings


def write_classified(X_test, y_pred, y_test):
    """
        Writes misclassfied examples to a file
    :param y_pred: Predicted labels
    :param y_test: True labels
    :param write_only_misclassified: if flag is set to True it writes only misclassified data
        else writes all the classified data
    """

    classified = {
        "sentence": [],
        "true_label": [],
        "predicted_label": [],
        "misclassified": []
    }

    for i, true_class in enumerate(y_test):
        classified["sentence"].append(X_test[i])
        classified["true_label"].append(LABELS[true_class])
        classified["predicted_label"].append(",".join([LABELS[pred] for pred in y_pred[i]]))

        is_misclassified = False
        if true_class not in y_pred[i]:
            is_misclassified = True
        classified["misclassified"].append(is_misclassified)

    df = pd.DataFrame(data=classified)
    df.to_csv(MISCLASSIFIED_FILE, index=None, header=True)


class LSTMClassifier(DataHandler):

    def __init__(self):
        DataHandler.__init__(self, TRAIN_TEST_SPLIT)
        self.epochs = EPOCHS
        self.embed_size = EMBEDDING_SIZE  # how big is each word vector
        self.embeddings = get_embeddings()
        self.max_features = 25000  # how many unique words to use (i.e num rows in embedding vector)
        self.maxlen = 100  # max number of words taken in a sentence

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

    def get_embedding_matrix(self):
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

    def train(self):
        # Get embedding feature
        embedding_matrix = self.get_embedding_matrix()

        # get model
        if INCLUDE_TYPE_FEATURE:
            model = self.model_with_type_feature(embedding_matrix)
            # Run model for some epochs
            model.fit([self.t_X_train, self.X_type_train], self.y_train, batch_size=256, epochs=self.epochs)
        else:
            model = self.model(embedding_matrix)
            # Run model for some epochs
            model.fit(self.t_X_train, self.y_train, batch_size=256, epochs=self.epochs)

        return model

    def model(self, embedding_matrix):
        """
            Bi-LSTM Keras model
        :return: model
        """
        words = embedding_matrix.shape[0]
        inp = Input(shape=(self.maxlen,))
        x = Embedding(words, self.embed_size, weights=[embedding_matrix], trainable=True)(inp)
        x = Bidirectional(LSTM(100, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(x)
        x = GlobalMaxPool1D()(x)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.25)(x)
        x = Dense(len(LABELS), activation="softmax")(x)

        model = Model(inputs=inp, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', top_k_accuracy])

        return model

    def model_with_type_feature(self, embedding_matrix):

        words = embedding_matrix.shape[0]
        inp = Input(shape=(self.maxlen,))
        x = Embedding(words, self.embed_size, weights=[embedding_matrix], trainable=True)(inp)
        lstm = Bidirectional(LSTM(100, dropout=0.25, recurrent_dropout=0.1))(x)
        inp_type = Input(shape=(16,))
        # pool = GlobalMaxPool1D()(lstm)
        conc = Concatenate()([lstm, inp_type])
        dense = Dense(100, activation="relu")(conc)
        drop = Dropout(0.25)(dense)
        acti = Dense(len(LABELS), activation="softmax")(drop)

        model = Model(inputs=[inp, inp_type], outputs=acti)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', top_k_accuracy])

        return model

    def predict(self, display=True):
        # TODO: would change after saving weights
        # train model
        model = self.train()

        # Predict scores
        if INCLUDE_TYPE_FEATURE:
            scores = model.evaluate([self.t_X_test, self.X_type_test], self.y_test, batch_size=100, verbose=1)
            y_pred = model.predict([self.t_X_test, self.X_type_test], batch_size=100, verbose=1)
        else:
            scores = model.evaluate(self.t_X_test, self.y_test, batch_size=100, verbose=1)
            y_pred = model.predict([self.t_X_test], batch_size=100, verbose=1)

        y_pred_classes = np.apply_along_axis(get_top_k_indices, 1, y_pred, K)
        y_test_class = np.argmax(self.y_test, axis=1)

        write_classified(self.X_test, y_pred_classes, y_test_class)

        if display:
            self.visualize(model, scores, y_test_class, y_pred_classes)

    def visualize(self, model, scores, y_test_class, y_pred_classes):
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print("top k accuracy - {}".format(scores[2] * 100))

        # plot_confusion_multiple(y_pred_classes, y_test_class, LABELS, K=K)
