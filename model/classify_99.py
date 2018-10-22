import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion

from utils.transformer import LengthVectorier

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