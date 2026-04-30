# Purpose: Clean the dataset, fix classification, handle duplicates, missing values, and outliers using Pandas
# Preprocess.py does the following
# Explore (26-28)
# Duplicates (25-28)
# Missing values (30-31)
# Outliers  (34-42)
# Bin quality
# Export to a new CSV that will be used for training


from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "winequality-red.csv"
DATA_PATH_CLEANED = BASE_DIR / "data" / "winequality-red-cleaned.csv"
DATA_PATH_NORMALIZED = BASE_DIR / "data" / "winequality-red-normalized.csv"

df = pd.read_csv(DATA_PATH, sep=";")
feature_columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]


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

# ===OUTLIERS USING IQR===
# Finding outliers in the dataset using IQR method 
for column in df.columns[:-1]:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower) & (df[column] <= upper)]

print(f"(Rows, Columns) after removing outliers as determined by IQR: {df.shape}") 


#===QUALITY BINNING===
# pd.cut(x, bins, labels)
df["quality_label"] = pd.cut(df["quality"], [2,4,6,8], labels=["low", "medium", "high"])
print(df["quality_label"].value_counts())


#===EXPORT TO NEW CSV===
# df.to_csv (new files name, maintain semicolon as seperator, do not add column and row nums)
df.to_csv(DATA_PATH_CLEANED, sep=';', index=False )

#===CORELATION MATRIX BETWEEN FEATURES===
# pd.set_option('display.max_columns', None)
# print(f"\nCorrelation Matrix Between Features:\n {df.drop(columns=["quality_label"]).corr()}")
# corr_matrix = df.drop(columns=["quality_label"]).corr()
# print (corr_matrix[abs(corr_matrix ) > 0.5 ].to_string())


#===NORMALIZING DATASET===
scaler = MinMaxScaler()
df[feature_columns] = scaler.fit_transform(df[feature_columns])
df.to_csv(DATA_PATH_NORMALIZED, sep=";", index=False)


















