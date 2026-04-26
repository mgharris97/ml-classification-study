# Purpose: Clean the dataset, fix classification, handle duplicates, missing values, and outliers using Pandas
# Preprocess.py does the following
# Explore (26-28)
# Duplicates (25-28)
# Missing values (30-31)
# Outliers 
# Bin quality
# Export to a new CSV that will be used for training


from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "winequality-red.csv"


df = pd.read_csv(DATA_PATH, sep=";")

# Explore the dataset to see if Pandas parses it correctly
print(f"total number of rows and columns in the dataset: {df.shape}")
print(df.dtypes) #return each column and its datatype either a float or int 
print(df.head()) # returns the first five rows

# ===DUPLICATE CHECK & REMOVAL===
print(f"Duplicated rows:  {df.duplicated().sum()}\n")
df = df.drop_duplicates()
print(f"(rows, columns): {df.shape}\n")

# ===EMPTY FIELD CHECK===
print(f"number of empty fields in each column:\n{df.isnull().sum()}")

# ===OUTLIERS===
for columns in df.columns[:-1]:
    # ill finish this later














