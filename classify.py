"""
    Classifies Political test
"""
from model.lstm_other import LSTMClassifier


if __name__ == '__main__':
    lstm_classifier = LSTMClassifier()
    lstm_classifier.predict()
