import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from model.classify_99 import classify_is_99
from model.classify_other import classify_all
from model.helper import load_embeddings, get_data, get_y_is_99, get_X_not_99, get_y_not_99

TRAIN_MODE = "train"
PREDICT_MODE = "predict"

MODEL_SAVE_DIR = "../weights/"


def _classify(X, y, embeddings, mode):

    # For training mode
    if mode == TRAIN_MODE:
        # Train 99 or not classfier
        classify_is_99(X, get_y_is_99(y), embeddings, mode)

        # Train classifier that classfies the rest of the labels
        classify_all(get_X_not_99(X, y), get_y_not_99(y), embeddings, mode)

    # For prediction mode
    elif mode == PREDICT_MODE:
        orig_X = X
        orig_y = y

        # get predictions of first classifier if label is 99 or not
        pred = classify_is_99(orig_X,  get_y_is_99(y), embeddings, mode)
        pred_labels = np.array(pred["pred"])

        # indices with predicted label as other than 99
        indices = np.argwhere(pred_labels==0)

        # indices with predicted label as 99
        indices_99 = np.argwhere(pred_labels==1)

        # The labels not 99 are classified by the second classifier
        X = orig_X[indices]
        y = orig_y[indices]
        pred_other = classify_all(X, y, embeddings, mode)

        # Create predicted by combining results from two classifiers
        y_pred = np.copy(orig_y)
        y_pred.put(indices_99, [99] * len(indices_99))
        y_pred.put(indices, pred_other)

        # print actual confusion matrix
        print(confusion_matrix(orig_y, y_pred))
        # print actual accuracy
        print(accuracy_score(orig_y, y_pred))


def classify():
    # TODO: Do cross validation
    embeddings = load_embeddings()
    data = get_data()
    # Split Data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(data["X"], data["y"], test_size=0.20, random_state=42)

    # Train
    _classify(X_train, y_train, embeddings, mode=TRAIN_MODE)
    # Test
    _classify(X_test, y_test, embeddings, mode=PREDICT_MODE)


if __name__ == '__main__':
    classify()
