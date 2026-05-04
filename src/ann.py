# Purpose: Artificial Neural Network classification on the wine quality dataset
# Dataset: winequality-red-normalized.csv
# Split: 75% training / 25% testing

from pathlib import Path
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH_NORMALIZED = BASE_DIR / "data" / "winequality-red-normalized.csv"

df = pd.read_csv(DATA_PATH_NORMALIZED, sep=";")

# Separate features and target
X = df.drop(columns=["quality", "quality_label"])
y = df["quality_label"]

# 75/25 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size:     {X_test.shape[0]}")
print()

# ===EXPERIMENT 1===
# Small network, fewer iterations
print("=" * 50)
print("EXPERIMENT 1: hidden_layer_sizes=(50,), max_iter=200")
print("=" * 50)
model1 = MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_train)
print(f"Training Accuracy: {accuracy_score(y_train, y_pred1):.4f}")
print(classification_report(y_train, y_pred1, zero_division=0))

# ===EXPERIMENT 2===
# Medium network, more iterations
print("=" * 50)
print("EXPERIMENT 2: hidden_layer_sizes=(100, 50), max_iter=500")
print("=" * 50)
model2 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_train)
print(f"Training Accuracy: {accuracy_score(y_train, y_pred2):.4f}")
print(classification_report(y_train, y_pred2, zero_division=0))

# ===EXPERIMENT 3===
# Larger network, more iterations, different activation
print("=" * 50)
print("EXPERIMENT 3: hidden_layer_sizes=(100, 100, 50), max_iter=1000, activation=tanh")
print("=" * 50)
model3 = MLPClassifier(hidden_layer_sizes=(100, 100, 50), max_iter=1000, activation="tanh", random_state=42)
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_train)
print(f"Training Accuracy: {accuracy_score(y_train, y_pred3):.4f}")
print(classification_report(y_train, y_pred3, zero_division=0))

# ===TESTING THE BEST MODEL===
# Apply best model to test set (change model3 to whichever performed best)
print("=" * 50)
print("TESTING BEST MODEL (Experiment 2) ON TEST SET")
print("=" * 50)
y_test_pred = model2.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(classification_report(y_test, y_test_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
