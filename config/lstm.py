from settings import WORD2VEC_FOLER, VISUALIZATION_FOLDER, DATA_FOLDER

# Enter all model related configurations here?

LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
          '12', '13', '14', '15', '16', '17', '18', '19', '20',
          '21', '99']

# Top k hyperparameter setting
K = 2

EPOCHS = 1
TRAIN_TEST_SPLIT = 0.2

EMBEDDING_FILE = WORD2VEC_FOLER / "glove.6B.100d.txt"
# EMBEDDING_FILE = "../word2vec/wiki-news-300d-1M.vec"

MISCLASSIFIED_FILE = VISUALIZATION_FOLDER / "misclassified-with-99-{}-{}-test.csv".format(K, "bush-clinton")

# Flag to see if test data should be imported or split
IMPORT_TEST_DATA = True
TRAINING_DATA_FILES = [DATA_FOLDER / "training_repository.csv", DATA_FOLDER / "benchmark_bushhw.csv"]
TEST_DATA_FILES = [DATA_FOLDER / "benchmark_clinton.csv"]
