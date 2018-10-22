import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

from model.helper import load_embeddings, get_data, get_y_is_99, get_X_not_99, get_y_not_99
from utils.transformer import WordEmbeddingVectorizer, LengthVectorier

TRAIN_MODE = "train"
PREDICT_MODE = "predict"

MODEL_SAVE_DIR = "../weights/"


def classify_is_99(X, y, embeddings, mode):
    """
    Classifies speech as 99 or not 99
    :return: {List} List of classifications
        1 if 99 and 0 if not
    """
    ngram_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    vectorizer = FeatureUnion([
        ("length", LengthVectorier()),
        ("ngram", ngram_vectorizer)
    ])

    pipe = Pipeline([
        ("features", vectorizer),
        ("classifier", LogisticRegression())
    ])

    if mode == TRAIN_MODE:
        # train the data
        pipe.fit(X, y)
        # save the model
        pickle.dump(pipe, open(MODEL_SAVE_DIR + "99.pickle", 'wb'))

        return

    elif mode == PREDICT_MODE:
        model = pickle.load(open(MODEL_SAVE_DIR + "99.pickle", 'rb'))
        y_pred = model.predict(X)
        conf_matrix = confusion_matrix(y, y_pred)
        print(conf_matrix)
        print(accuracy_score(y, y_pred))

        return {
            "pred": y_pred
        }


def classify_all(X, y, embeddings, mode):
    bag_of_words_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 9), stop_words='english')
    wordVectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')

    vectorizer = FeatureUnion([
        ("ngrams", wordVectorizer),
        ("w2v vectorizer", WordEmbeddingVectorizer(embeddings)),
    ])
    classifier = LogisticRegression()

    pipe = Pipeline([
        ('vectorizer', vectorizer),
        ("classifier", classifier)
    ])

    # kf = KFold(n_splits=5, shuffle=True)
    #
    # # Kflod validation on the data
    #
    # for train_index, test_index in kf.split(X):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     pipe.fit(X_train, y_train)
    #     y_pred = pipe.predict(X_test)
    #     conf_matrix = confusion_matrix(y_test, y_pred)
    #     print(accuracy_score(y_test, y_pred))
    #     print(conf_matrix)

    if mode == TRAIN_MODE:
        # train the data
        pipe.fit(X, y)

        # save the model
        pickle.dump(pipe, open(MODEL_SAVE_DIR + "all.pickle", 'wb'))
        return

    elif mode == PREDICT_MODE:
        model = pickle.load(open(MODEL_SAVE_DIR + "all.pickle", 'rb'))
        X_transformed = vectorizer.fit_transform(X)
        y_pred = model.predict(X_transformed)

        # Confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)
        print(conf_matrix)
        print(accuracy_score(y, y_pred))

        return {
            "pred": y_pred
        }


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
