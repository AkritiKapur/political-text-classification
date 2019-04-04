"""
    Performs multi-task learning using a uniform LSTM model architecture
"""
import pandas as pd
import numpy as np
import keras

from keras import Input, Model
from keras.layers import Dense, Embedding, LSTM, Bidirectional, GlobalMaxPool1D, Dropout, Concatenate
from keras.metrics import top_k_categorical_accuracy

from sklearn.model_selection import train_test_split

from config.lstm import EPOCHS, TRAIN_TEST_SPLIT, EMBEDDING_SIZE, LABELS, K
from model.helper import get_one_hot
from settings import WORD2VEC_FOLER, VISUALIZATION_FOLDER, DATA_FOLDER
from utils.embedding import Embedding

TRAINING_DATA_FILES = {
    "Reagan": DATA_FOLDER / "training_repository.csv",
    "Clinton": DATA_FOLDER / "benchmark_clinton.csv",
    "Bush": DATA_FOLDER / "benchmark_bushhw.csv"
}


def top_k_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=K)


def _get_data(f_key):
    df = pd.read_csv(TRAINING_DATA_FILES[f_key], encoding="ISO-8859-1")
    X = df['Sentences'].values
    y = df['pap_fin'].values

    data = {
        "X": np.array(X),
        "y": np.array(y),
        "y_name": np.array([f_key] * len(X))
    }

    return data


def get_data(files):
    fin_data = {
        "X": np.array([]),
        "y": np.array([]),
        "y_name": np.array([])
    }

    for f in files:
        data = _get_data(f)
        fin_data["X"] = np.concatenate((fin_data["X"], data["X"]), axis=None)
        fin_data["y"] = np.concatenate((fin_data["y"], data["y"]), axis=None)
        fin_data["y_name"] = np.concatenate((fin_data["y_name"], data["y_name"]), axis=None)

    return fin_data


def get_train_test_split(data, percent_split=0.2):
    X = data["X"]
    y_topic = get_one_hot(data["y"])
    y_name = get_one_hot(data["y_name"])

    X_train, X_test, \
    y_name_train, y_name_test, \
    y_topic_train, y_topic_test = train_test_split(X, y_name, y_topic,
                                                   test_size=percent_split, random_state=1435)

    return X_train, X_test, y_name_train, y_name_test, y_topic_train, y_topic_test


class MultiTaskClassifier:
    def __init__(self, data):
        X_train, X_test, \
        self.y_name_train, self.y_name_test, \
        self.y_topic_train, self.y_topic_test = get_train_test_split(data, TRAIN_TEST_SPLIT)

        self.max_features = 25000  # how many unique words to use (i.e num rows in embedding vector)
        self.maxlen = 100
        self.epochs = EPOCHS
        self.embed_size = EMBEDDING_SIZE

        embedding = Embedding(X_train, X_test, self.max_features, self.maxlen)
        self.embedding_matrix = embedding.generate_embedding_matrix()
        self.X_train, self.X_test = embedding.t_X_train, embedding.t_X_test

    def train(self):
        model = self.get_model()
        # Run model for some epochs
        model.fit(self.X_train, [self.y_name_train, self.y_topic_train], batch_size=256, epochs=self.epochs)

        return model

    def predict(self):
        trained_model = self.train()
        scores = trained_model.evaluate(self.X_test, [self.y_name_test, self.y_topic_test], batch_size=100, verbose=1)
        y_pred = trained_model.predict([self.X_test], batch_size=100, verbose=1)

        print("%s: %.2f%%" % (trained_model.metrics_names[1], scores[1] * 100))
        print("%s: %.2f%%" % (trained_model.metrics_names[2], scores[2] * 100))
        pass

    def get_model(self):
        words = self.embedding_matrix.shape[0]
        inp = Input(shape=(self.maxlen,))
        x = keras.layers.Embedding(words, self.embed_size, weights=[self.embedding_matrix], trainable=True)(inp)
        x = Bidirectional(LSTM(100, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(x)
        x = GlobalMaxPool1D()(x)
        x = Dense(100, activation="relu")(x)
        drop_layer = Dropout(0.25)(x)

        out1 = Dense(3, activation="softmax", name="name")(drop_layer)
        out2 = Dense(len(LABELS), activation="softmax", name="topic")(drop_layer)

        model = Model(inputs=inp, outputs=[out1, out2])
        model.compile(optimizer='adam', loss=['categorical_crossentropy', 'categorical_crossentropy'], metrics=['acc'])

        return model

    def get_loss(self):
        pass


if __name__ == '__main__':
    data = get_data(TRAINING_DATA_FILES)
    multiTaskClassifier = MultiTaskClassifier(data)
    multiTaskClassifier.predict()
