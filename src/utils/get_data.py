import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.containts.containts import DATA_FILE


def load_vocabulary(path=DATA_FILE):
    with open(path, 'r', encoding='utf-8') as file:
        voca = [line.strip() for line in file if line.strip()]
    return voca
