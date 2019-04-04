"""
    Performs multi-task learning using a uniform LSTM model architecture
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from model.helper import get_one_hot
from settings import WORD2VEC_FOLER, VISUALIZATION_FOLDER, DATA_FOLDER
from utils.embedding import Embedding

TRAIN_TEST_SPLIT = 0.2

TRAINING_DATA_FILES = {
    "Reagan": DATA_FOLDER / "training_repository.csv",
    "Clinton": DATA_FOLDER / "benchmark_clinton.csv",
    "Bush": DATA_FOLDER / "benchmark_bushhw.csv"
}


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
        y_name_train, y_name_test, \
        y_topic_train, y_topic_test = get_train_test_split(data, TRAIN_TEST_SPLIT)

        self.max_features = 25000  # how many unique words to use (i.e num rows in embedding vector)
        self.maxlen = 100

        embedding = Embedding(X_train, X_test, self.max_features, self.maxlen)
        self.embedding_matrix = embedding.generate_embedding_matrix()



def get_loss():
    pass


def train(X_train, X_test, y_topic, y_pres):
    # Get embedding feature
    embedding_matrix = get_embedding_matrix()


def test():
    pass


def get_model():
    pass


if __name__ == '__main__':
    data = get_data(TRAINING_DATA_FILES)






