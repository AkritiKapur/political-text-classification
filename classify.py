"""
    Classifies Political test
"""

# List all your configurations under config

from model.lstm_other import LSTMClassifier


if __name__ == '__main__':
    lstm_classifier = LSTMClassifier()
    lstm_classifier.predict()
