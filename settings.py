"""
    Add all project related settings here!
"""

import os
from pathlib import Path

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = Path("{}/data/".format(PROJECT_DIR))
VISUALIZATION_FOLDER = Path("{}/visualizations/".format(PROJECT_DIR))
WORD2VEC_FOLER = Path("{}/word2vec/".format(PROJECT_DIR))
