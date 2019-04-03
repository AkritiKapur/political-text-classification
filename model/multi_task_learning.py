"""
    Performs multi-task learning using a uniform LSTM model architecture
"""
import pandas as pd
import numpy as np
from settings import WORD2VEC_FOLER, VISUALIZATION_FOLDER, DATA_FOLDER

EMBEDDING_SIZE = 200
TRAINING_DATA_FILES = {
    "Reagan": DATA_FOLDER / "training_repository.csv",
    "Clinton": DATA_FOLDER / "benchmark_clinton.csv",
    "Bush": DATA_FOLDER / "benchmark_bushhw.csv"
}

EMBEDDING_FILE = WORD2VEC_FOLER / "glove.6B.{}d.txt".format(EMBEDDING_SIZE)


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


def get_loss():
    pass


def train():
    pass


def test():
    pass


def get_model():
    pass


if __name__ == '__main__':
    data = get_data(TRAINING_DATA_FILES)

