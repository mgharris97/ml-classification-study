from pathlib import Path
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH_NORMALIZED = BASE_DIR / "data" / "winequality-red-normalized.csv"

df = pd.read_csv(DATA_PATH_NORMALIZED, sep=";")
