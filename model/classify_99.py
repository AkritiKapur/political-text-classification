import pickle
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion

from config.lstm import TRAIN_TEST_SPLIT
from exceptions import TrainLabelException
from model.data_handler import DataHandler
from utils.transformer import LengthVectorier

TRAIN_MODE = "train"
PREDICT_MODE = "predict"

MODEL_SAVE_DIR = "../weights/"


class Classifier_99(DataHandler):

    def __init__(self, X, y):
        DataHandler.__init__(self, TRAIN_TEST_SPLIT)
        self.classifier = LogisticRegression()
        self.X_train = X
        self.y_train = y

    def get_features(self):
        feature_vectorizer = FeatureUnion(
            [
                ("length", self.get_length_feature()),
                ("ngram", self.get_ngram_feature())
            ]
        )

        return feature_vectorizer

    def get_length_feature(self):
        return LengthVectorier()

    def get_ngram_feature(self):
        return TfidfVectorizer(ngram_range=(1, 2), stop_words='english')

    def train(self):
        pipe = Pipeline([
            ("features", self.get_features()),
            ("classifier", self.classifier)
        ])

        pipe.fit(self.X_train, self.y_train)

        return pipe

    def predict(self, X, y=None, display_stats=True):
        model = self.train()
        y_pred = model.predict(X)

        if display_stats and y:
            self.get_accuracy_stats(y, y_pred)

    def get_accuracy_stats(self, y, y_pred):
        print("Accuracy of test set is {}", accuracy_score(y, y_pred))


if __name__ == '__main__':

    classifier = Classifier_99()

#
# def classify_is_99(X, y, embeddings, mode):
#     """
#     Classifies speech as 99 or not 99
#     :return: {List} List of classifications
#         1 if 99 and 0 if not
#     """
#     ngram_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
#     vectorizer = FeatureUnion([
#         ("length", LengthVectorier()),
#         ("ngram", ngram_vectorizer)
#     ])
#
#     pipe = Pipeline([
#         ("features", vectorizer),
#         ("classifier", LogisticRegression())
#     ])
#
#     if mode == TRAIN_MODE:
#         # train the data
#         pipe.fit(X, y)
#         # save the model
#         pickle.dump(pipe, open(MODEL_SAVE_DIR + "99.pickle", 'wb'))
#
#         return
#
#     elif mode == PREDICT_MODE:
#         model = pickle.load(open(MODEL_SAVE_DIR + "99.pickle", 'rb'))
#         y_pred = model.predict(X)
#         conf_matrix = confusion_matrix(y, y_pred)
#         print(conf_matrix)
#         print(accuracy_score(y, y_pred))
#
#         return {
#             "pred": y_pred
#         }
