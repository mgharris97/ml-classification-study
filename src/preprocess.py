# Purpose: Clean the dataset and fix classification using Pandas
# Pathlib: Used to create a relative filepath regardless of OS
# Note: The dataset uses semicolons (;) as a seperator not commas (,)

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "winequality-red.csv"

df = pd.read_csv(DATA_PATH, sep=";")

# Explore the dataset to see if Pandas parses it correctly
print(df.shape) #print number of columns and rows
print(df.dtypes) #return each column and its datatype
print(df.head()) # returns the first five rows




