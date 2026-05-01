from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH_NORMALIZED = BASE_DIR / "data" / "winequality-red-normalized.csv"

df = pd.read_csv(DATA_PATH_NORMALIZED, sep=";")

# continue here
