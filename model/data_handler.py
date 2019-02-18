import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config.lstm import TRAINING_DATA_FILES, TEST_DATA_FILES, IMPORT_TEST_DATA, \
    INCLUDE_TYPE_FEATURE
from model.helper import import_data


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


class DataHandler:
    def __init__(self, percent_split):
        data = import_data(TRAINING_DATA_FILES)
        self.X_train, self.X_type_train, self.y_train, self.X_test, self.X_type_test, self.y_test = \
            self.split_data(data, percent_split)

    def split_data(self, data, percent):
        """
            Splits data into train and test data
        :param percent: test percent out of the whole data.
        :return {dict} X train, X test, y train and y test
        """
        X_type_train = None
        X_type_test = None
        X = data["X"]
        # TODO: fix one hot conversion
        y = get_one_hot(data["y"])
        # y = data["y"]

        if not IMPORT_TEST_DATA:

            if INCLUDE_TYPE_FEATURE:
                X_type = data["X_type"]
                X_type = get_one_hot(X_type)
                X_train, X_test, X_type_train, X_type_test, y_train, y_test = train_test_split(X, X_type, y,
                                                                                               test_size=percent,
                                                                                               random_state=1435)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percent, random_state=1435)
        else:
            # Test data set should be imported
            X_train, y_train = X, y
            test_data = import_data(TEST_DATA_FILES)

            X_test, y_test = test_data["X"], test_data["y"]

            # Permute test data
            perm = np.random.permutation(X_test.shape[0])
            X_test = X_test[perm]
            y_test = get_one_hot(y_test[perm])

        return X_train, X_type_train, y_train, X_test, X_type_test, y_test
