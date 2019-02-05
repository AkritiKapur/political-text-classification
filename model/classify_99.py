from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion

from config.lstm import TRAIN_TEST_SPLIT
from model.data_handler import DataHandler
from model.helper import load_embeddings, get_y_is_99
from utils.transformer import LengthVectorier, WordEmbeddingVectorizer

TRAIN_MODE = "train"
PREDICT_MODE = "predict"

MODEL_SAVE_DIR = "../weights/"


class Classify99(DataHandler):
    def __init__(self):
        # intialize train test set
        DataHandler.__init__(self, TRAIN_TEST_SPLIT)
        self.y_train = self.restructure_labels(self.y_train)
        self.y_test = self.restructure_labels(self.y_test)

    def restructure_labels(self, y):
        """
            Restructure labels into 99s and others
        :return: restructures lablels where 1 indicates 99 and 0 indicates other labels
        """

        return get_y_is_99(y)


class BaselineClassifier(DataHandler):

    def __init__(self):
        DataHandler.__init__(self, TRAIN_TEST_SPLIT)
        self.classifier = LogisticRegression()
        self.embeddings = load_embeddings()

    def get_features(self):
        feature_vectorizer = FeatureUnion(
            [
                ("length", self.get_length_feature()),
                ("ngram", self.get_ngram_feature()),
                ("w2v vectorizer", WordEmbeddingVectorizer(self.embeddings))
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

    def predict(self, display_stats=True):
        model = self.train()
        y_pred = model.predict(self.X_test)

        if display_stats:
            self.get_accuracy_stats(self.y_test, y_pred)

    def get_accuracy_stats(self, y, y_pred):
        print("Accuracy of test set is {}".format(accuracy_score(y, y_pred)))


if __name__ == '__main__':

    classifier = BaselineClassifier()
    classifier.predict()
