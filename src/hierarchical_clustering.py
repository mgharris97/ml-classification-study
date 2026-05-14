# Purpose: Hierarchical clustering on the wine quality dataset
# Dataset: winequality-red-normalized.csv
# Note: Uses Ward linkage with Euclidean distance to match Orange defaults

from pathlib import Path
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH_NORMALIZED = BASE_DIR / "data" / "winequality-red-normalized.csv"

df = pd.read_csv(DATA_PATH_NORMALIZED, sep=";")

# Separate features from target
feature_cols = [c for c in df.columns if c not in ("quality", "quality_label")]

# Use a 200-row sample so the dendrogram is readable
SAMPLE_SIZE = 200
df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
X = df_sample[feature_cols].values

print(f"Sample size:   {SAMPLE_SIZE}")
print(f"Features used: {feature_cols}")
print()

# Linkage matrix is shared by all three experiments (only the cut changes)
Z = linkage(X, method="ward", metric="euclidean")


def run_experiment(exp_num, n_clusters):
    """Fit AgglomerativeClustering, plot the dendrogram with a cut line, and return labels."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", metric="euclidean")
    labels = model.fit_predict(X)

    # Compute the distance threshold corresponding to n_clusters
    # (Z[-(n_clusters - 1), 2] is the merge distance that produces n_clusters)
    cutoff = Z[-(n_clusters - 1), 2] - 1e-6 if n_clusters > 1 else Z[-1, 2] + 1

    fig, ax = plt.subplots(figsize=(18, 6))
    dendrogram(Z, ax=ax, color_threshold=cutoff,
               above_threshold_color="gray", no_labels=True)
    ax.axhline(y=cutoff, color="red", linestyle="--", linewidth=2,
               label=f"Cut for {n_clusters} clusters")
    ax.set_title(f"Experiment {exp_num} - Dendrogram (Ward linkage, {n_clusters} clusters)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Distance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / f"dendrogram_exp{exp_num}.png", dpi=150)
    plt.show()

    print(f"[Experiment {exp_num}] n_clusters = {n_clusters}")
    return labels


# ===EXPERIMENT 1===
# Coarse: 2 clusters
print("=" * 50)
print("EXPERIMENT 1: n_clusters=2 (coarse)")
print("=" * 50)
labels_1 = run_experiment(1, n_clusters=2)
print()

# ===EXPERIMENT 2===
# Matches 3-class quality binning (low/medium/high)
print("=" * 50)
print("EXPERIMENT 2: n_clusters=3 (matches 3-class binning)")
print("=" * 50)
labels_2 = run_experiment(2, n_clusters=3)
print()

# ===EXPERIMENT 3===
# Finer: 6 clusters
print("=" * 50)
print("EXPERIMENT 3: n_clusters=6 (fine)")
print("=" * 50)
labels_3 = run_experiment(3, n_clusters=6)
print()

# ===CLUSTER ANALYSIS (BEST EXPERIMENT)===
# Experiment 2 selected: 3 clusters aligns with low/medium/high quality binning
print("=" * 50)
print("CLUSTER ANALYSIS - Experiment 2 (3 clusters)")
print("=" * 50)
df_sample["cluster"] = labels_2

print("Cluster vs Quality Label distribution:")
print(pd.crosstab(df_sample["cluster"], df_sample["quality_label"]))
print()

# Scatter plot: Alcohol vs Volatile Acidity, side-by-side cluster vs true label
color_map = {0: "purple", 1: "gold", 2: "steelblue", 3: "tomato", 4: "green", 5: "orange"}
qual_colors = {"low": "tomato", "medium": "steelblue", "high": "green"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for c in sorted(df_sample["cluster"].unique()):
    mask = df_sample["cluster"] == c
    axes[0].scatter(df_sample.loc[mask, "alcohol"],
                    df_sample.loc[mask, "volatile acidity"],
                    c=color_map.get(c, "gray"), alpha=0.6, s=30, label=f"Cluster {c}")
axes[0].set_xlabel("Alcohol (normalized)")
axes[0].set_ylabel("Volatile Acidity (normalized)")
axes[0].set_title("Hierarchical Clusters (n=3)")
axes[0].legend()

for q, color in qual_colors.items():
    mask = df_sample["quality_label"] == q
    axes[1].scatter(df_sample.loc[mask, "alcohol"],
                    df_sample.loc[mask, "volatile acidity"],
                    c=color, alpha=0.6, s=30, label=q)
axes[1].set_xlabel("Alcohol (normalized)")
axes[1].set_ylabel("Volatile Acidity (normalized)")
axes[1].set_title("True Quality Labels")
axes[1].legend()

plt.suptitle("Hierarchical Clustering vs True Labels", fontsize=13)
plt.tight_layout()
plt.savefig(BASE_DIR / "hierarchical_cluster_scatter.png", dpi=150)
plt.show()