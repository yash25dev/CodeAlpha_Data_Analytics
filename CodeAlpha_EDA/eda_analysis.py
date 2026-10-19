
# CodeAlpha Internship - Task 2
# Exploratory Data Analysis (EDA)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# 1. Load the dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Replace numeric target labels with flower names
df["species"] = df["target"].map(
    dict(enumerate(iris.target_names))
)
df.drop(columns=["target"], inplace=True)

# Create output folder
os.makedirs("graphs", exist_ok=True)

# Save dataset
df.to_csv("iris_dataset.csv", index=False)

print("\n========== DATASET OVERVIEW ==========")
print("First five rows:\n", df.head())
print("\nDataset shape:", df.shape)
print("\nColumn names:\n", df.columns.tolist())
print("\nData types:\n", df.dtypes)

# 2. Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# 3. Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 4. Check duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print("Duplicate rows:", df.duplicated().sum())

# 5. Count each species
print("\n========== SPECIES DISTRIBUTION ==========")
print(df["species"].value_counts())

# 6. Compare average measurements by species
print("\n========== SPECIES-WISE AVERAGES ==========")
print(df.groupby("species").mean(numeric_only=True))

# 7. Correlation analysis
numeric_df = df.select_dtypes(include=np.number)
correlation = numeric_df.corr()

print("\n========== CORRELATION MATRIX ==========")
print(correlation.round(2))

# 8. Graph 1: Species count
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="species")
plt.title("Number of Flowers in Each Species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("graphs/species_count.png")
plt.close()

# 9. Graph 2: Feature distributions
numeric_df.hist(figsize=(10, 8), bins=15)
plt.suptitle("Distribution of Iris Measurements")
plt.tight_layout()
plt.savefig("graphs/feature_distribution.png")
plt.close()

# 10. Graph 3: Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("graphs/correlation_heatmap.png")
plt.close()

# 11. Graph 4: Compare measurements by species
plt.figure(figsize=(9, 6))
sns.boxplot(
    data=df,
    x="species",
    y="petal length (cm)"
)
plt.title("Petal Length by Iris Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.savefig("graphs/species_comparison.png")
plt.close()

# 12. Identify the strongest positive correlation
corr_pairs = correlation.where(
    np.triu(np.ones(correlation.shape), k=1).astype(bool)
).stack()

strongest_pair = corr_pairs.idxmax()

print("\n========== KEY FINDING ==========")
print("Strongest positive correlation:")
print(strongest_pair[0], "and", strongest_pair[1])
print("Correlation:", round(corr_pairs.max(), 3))

print("\nEDA completed successfully!")
print("Dataset saved as iris_dataset.csv")
print("All four graphs saved in the graphs folder.")
