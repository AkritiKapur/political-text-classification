"""
took help from: https://www.kaggle.com/snlpnkj/bidirectional-lstm-keras,
                https://www.kaggle.com/liliasimeonova/sentiment-analysis-with-bidirectional-lstm

others - 16426
"""
import functools
import sys

import keras
import pandas as pd

import numpy as np
from keras import Input, Model
from keras.layers import Dense, Embedding, LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from exceptions import TestInputException
from model.helper import get_data, get_X_not_99, get_y_not_99

# Set parameters
from settings import VISUALIZATION_FOLDER, WORD2VEC_FOLER, DATA_FOLDER
from utils.plot import plot_confusion_matrix, plot_confusion_matrix_blue
from utils.process_data import get_top_k_indices

embed_size = 100  # how big is each word vector
max_features = 25000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100  # max number of words in a comment to use

LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
          '12', '13', '14', '15', '16', '17', '18', '19', '20',
          '21', '99']

# Top k hyperparameter setting
K = 2

EPOCHS = 9

EMBEDDING_FILE =  WORD2VEC_FOLER / "glove.6B.100d.txt"
# EMBEDDING_FILE = "../word2vec/wiki-news-300d-1M.vec"

MISCLASSIFIED_FILE = VISUALIZATION_FOLDER / "misclassified-with-99-{}.csv".format(K)

# Flag to see if test data should be imported or split
IMPORT_TEST_DATA = True
TEST_DATA_FILES = [DATA_FOLDER / "benchmark_bushhw.csv", DATA_FOLDER / "benchmark_clinton.csv"]


def get_test_data():
    try:
        if not TEST_DATA_FILES:
            raise TestInputException("No test data input files specified")
    except TestInputException as te:
        sys.exit()
    else:
        pass


def top_k_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=K)


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def pre_process(X_train, X_test):
    """
        Handles preprocessing of data
    :return:
    """

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train))
    list_tokenized_train = tokenizer.texts_to_sequences(X_train)
    list_tokenized_test = tokenizer.texts_to_sequences(X_test)
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

    return X_t, X_te, tokenizer


def get_one_hot(y):
    # Help: https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    return dummy_y


def get_embeddings():
    embeddings = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
    return embeddings


def write_misclassified(X_test, y_pred, y_test):
    """
        Writes misclassfied examples to a file
    :param y_pred: Predicted labels
    :param y_test: True labels
    """

    misclassified = {
        "sentence": [],
        "true_label": [],
        "predicted_label": []
    }

    for i, true_class in enumerate(y_test):
        if true_class not in y_pred[i]:
            misclassified["sentence"].append(X_test[i])
            misclassified["true_label"].append(LABELS[true_class])
            misclassified["predicted_label"].append(",".join([LABELS[pred] for pred in y_pred[i]]))

    df = pd.DataFrame(data=misclassified)
    df.to_csv(MISCLASSIFIED_FILE, index=None, header=True)


if __name__ == '__main__':
    data = get_data()

    X = data["X"]
    y = data["y"]

    # # filter 99s
    # X = get_X_not_99(X, y)
    # y = get_y_not_99(y)

    # One-hot-encodings
    y = get_one_hot(y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1435)
    X_test_untokenized = X_test

    X_train, X_test, tokenizer = pre_process(X_train, X_test)

    embeddings = get_embeddings()

    all_embs = np.stack(list(embeddings.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words+1, embed_size))

    for word, i in word_index.items():
        # TODO: replace this with stop words and frequency instead
        if i >= max_features:
            continue
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    inp = Input(shape=(maxlen,))
    x = Embedding(nb_words+1, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    x = Bidirectional(LSTM(100, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(len(LABELS), activation="softmax")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', top_k_accuracy])
    model.fit(X_train, Y_train, batch_size=256, epochs=EPOCHS)

    # Prediction
    scores = model.evaluate(X_test, Y_test, batch_size=100, verbose=1)

    y_pred = model.predict([X_test], batch_size=100, verbose=1)
    y_pred_classes = np.apply_along_axis(get_top_k_indices, 1, y_pred, K)
    y_test_class = np.argmax(Y_test, axis=1)

    write_misclassified(X_test_untokenized, y_pred_classes, y_test_class)

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("top k accuracy - {}".format(scores[2] * 100))


    # TODO: make confusion matrix configurable for different K values
    if K == 1:
        print('Confusion Matrix')
        cm = confusion_matrix(y_pred_classes.flatten(), y_test_class)
        print(cm)
        plot_confusion_matrix(cm, LABELS)
        plt = plot_confusion_matrix_blue(cm, LABELS)
        plt.show()
