# this where the dataset will be cleaned and classes 3-8 will be binned into
# three classes (low, medium, high)
# the purpos of preprocess is to get thedataset ready for training / 

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "winequalityred.csv"



