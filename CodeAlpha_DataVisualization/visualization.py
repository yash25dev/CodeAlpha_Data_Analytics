
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine

# Use paths relative to this Python file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Load a different dataset: Wine Classification
wine = load_wine(as_frame=True)
df = wine.data.copy()

df["target"] = wine.target
df["class_name"] = df["target"].map(
    {i: f"Class {i}" for i in range(len(wine.target_names))}
)

# Save dataset as CSV
csv_path = os.path.join(BASE_DIR, "wine_dataset.csv")
df.to_csv(csv_path, index=False)

print("Wine dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nDataset saved to:", csv_path)


def save_graph(filename):
    """Save the current graph in the graphs folder."""
    plt.tight_layout()
    plt.savefig(
        os.path.join(GRAPHS_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    print("Created graph:", filename)


# GRAPH 1: Number of samples in each wine class
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="class_name")
plt.title("Distribution of Wine Classes")
plt.xlabel("Wine Class")
plt.ylabel("Number of Samples")
save_graph("01_class_distribution.png")


# GRAPH 2: Correlation heatmap of all 13 features
plt.figure(figsize=(13, 10))
correlation = df[wine.feature_names].corr()
sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".1f"
)
plt.title("Correlation Heatmap of Wine Features")
save_graph("02_correlation_heatmap.png")


# GRAPH 3: Distribution of four selected features
selected_features = [
    "alcohol",
    "malic_acid",
    "color_intensity",
    "proline"
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for feature, ax in zip(selected_features, axes.flatten()):
    sns.histplot(
        data=df,
        x=feature,
        hue="class_name",
        kde=True,
        ax=ax
    )
    ax.set_title(f"Distribution of {feature.replace('_', ' ').title()}")

fig.suptitle("Distribution of Selected Wine Features", fontsize=15)
plt.tight_layout()
fig.savefig(
    os.path.join(GRAPHS_DIR, "03_feature_distributions.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)
print("Created graph: 03_feature_distributions.png")


# GRAPH 4: Relationship between alcohol and color intensity
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="alcohol",
    y="color_intensity",
    hue="class_name",
    s=80
)
plt.title("Alcohol vs Color Intensity by Wine Class")
plt.xlabel("Alcohol")
plt.ylabel("Color Intensity")
save_graph("04_scatter_plot.png")


# GRAPH 5: Compare alcohol values across wine classes
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="class_name",
    y="alcohol"
)
plt.title("Alcohol Content Across Wine Classes")
plt.xlabel("Wine Class")
plt.ylabel("Alcohol")
save_graph("05_box_plot.png")


print("\nVisualization completed successfully!")
print("Dataset: wine_dataset.csv")
print("All five graphs are saved in the graphs folder.")
