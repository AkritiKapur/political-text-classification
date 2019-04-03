import sys

import gensim
import pandas as pd
import numpy as np

################### WORD EMBEDDINGS ######################
from config.lstm import INCLUDE_TYPE_FEATURE
from exceptions import InputException


def load_embeddings():
    """
        Loads gensim embeddings
    :return: word2vec matrix
    """
    model = gensim.models.KeyedVectors.load_word2vec_format('../word2vec/GoogleNews-vectors-negative300.bin',
                                                            binary=True)
    return model

################################## DATA #################################


def is_99(row):
    """
        Classifies pandas row column "pap_fin" as 99 or not 99
    :param row:
    :return:
        1  -  row is 99
        0  -  row is not 99
    """
    if row['pap_fin'] == 99:
        return 1
    else:
        return 0


def import_data(data_files):
    try:
        if not data_files:
            raise InputException("No import files specified")
    except InputException as te:
        sys.exit()
    else:
        if INCLUDE_TYPE_FEATURE:
            return get_all_data_with_type(data_files)
        return get_all_data(data_files)


def get_data(f, get_type_feature=False):
    """
        Load data from CSV
    :return: Training data X and labels y
    """
    df = pd.read_csv(f, encoding="ISO-8859-1")
    X = df['Sentences'].values
    y = df['pap_fin'].values

    data = {
        "X": np.array(X),
        "y": np.array(y)
    }

    if get_type_feature:
        X_type = df['Statement.Type'].values
        data["X_type"] = np.array(X_type)

    return data


def get_all_data(files):
    """
        Load data from CSVs
    :param f: {list} files for which data needs to be extracted into a dataframe
    :return: data X and labels y
    """
    fin_data = {
        "X": np.array([]),
        "y": np.array([])
    }

    for f in files:
        data = get_data(f)
        fin_data["X"] = np.concatenate((fin_data["X"], data["X"]), axis=None)
        fin_data["y"] = np.concatenate((fin_data["y"], data["y"]), axis=None)

    return fin_data


def get_all_data_with_type(files):
    fin_data = {
        "X": np.array([]),
        "y": np.array([]),
        "X_type": np.array([])
    }

    for f in files:
        data = get_data(f, True)
        fin_data["X"] = np.concatenate((fin_data["X"], data["X"]), axis=None)
        fin_data["y"] = np.concatenate((fin_data["y"], data["y"]), axis=None)
        fin_data["X_type"] = np.concatenate((fin_data["X_type"], data["X_type"]), axis=None)

    return fin_data


def get_y_is_99(y):
    # filter y to indicate which is 99 and which isn't
    y_is_99 = y == 99
    return y_is_99.astype(int)


def get_X_not_99(X, y):
    return X[y != 99]


def get_y_not_99(y):
    return y[y != 99]


def get_one_hot(y):
    """
        Converts labels into one hot vector
    :param y: {List} labels
    :return: {2D array} one hot labels
    """
    # Help: https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    return dummy_y


################################ SAVE WEIGHTS ###################################


def save_weights(weights):
    pass
