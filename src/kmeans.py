from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH_NORMALIZED = BASE_DIR / "data" / "winequality-red-normalized.csv"

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df2 = pd.read_csv(DATA_PATH_NORMALIZED, sep=";")

X = df2.drop(["quality", "quality_label"], axis=1)

print(X.shape)
print(X.columns)



# -----------------------------
# 2. SILHOUETTE SCORES
# -----------------------------
results = []

for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=8745, n_init=10)
    clusters = kmeans.fit_predict(X)

    score = silhouette_score(X, clusters)
    results.append((k, score))

    print(k, score)

# -----------------------------
# 3. BEST MODEL SELECTION
# -----------------------------
best_k = max(results, key=lambda x: x[1])[0] #put in order by score

kmeans = KMeans(n_clusters=best_k, random_state=8745, n_init=10)
clusters = kmeans.fit_predict(X)

df2["cluster"] = clusters
centroids = kmeans.cluster_centers_

print(best_k)
print(len(clusters))

# -----------------------------
# 4. VISUALISATION - CLUSTERS + CENTROIDS
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# ---- Plot 1: clusters ----
axes[0].scatter(
    df2["alcohol"],
    df2["volatile acidity"],
    c=clusters,
    cmap="viridis",
)

axes[0].set_xlabel("Alcohol")
axes[0].set_ylabel("Volatile Acidity")
axes[0].set_title(f"K-Means Clusters (k={best_k})")

# get right feature index from X data
alcohol_idx = X.columns.get_loc("alcohol")
volatile_idx = X.columns.get_loc("volatile acidity")


axes[0].scatter( # plot centroids in the graph
    centroids[:, alcohol_idx],
    centroids[:, volatile_idx],
    color="red",
    s=150,
    marker="X",
    label="Centroids"
)

axes[0].legend()

# ----Plot 2: silhouette score based on k

#convert result to dataframe so its easy to use with plt
results_df = pd.DataFrame(results, columns=["k", "Silhouette Score"]) 

axes[1].plot(
    results_df["k"],
    results_df["Silhouette Score"],
    marker="o"
)

axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score for Different k Values")

axes[1].grid(True)

plt.tight_layout()
plt.show()



# -----------------------------
# RESULTS ANALYSIS
# -----------------------------

# TABLE OF RESULTS

results_df = pd.DataFrame(results, columns=["k", "Silhouette Score"])

print("\nSilhouette Score Results:")
print(results_df)

# CLUSTER vs TRUE QUALITY TABLE

print("\nCluster vs Quality Distribution:")
print(pd.crosstab(df2["cluster"], df2["quality"]))

