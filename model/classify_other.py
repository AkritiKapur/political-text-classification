import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion

from utils.transformer import WordEmbeddingVectorizer

TRAIN_MODE = "train"
PREDICT_MODE = "predict"

MODEL_SAVE_DIR = "../weights/"


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
