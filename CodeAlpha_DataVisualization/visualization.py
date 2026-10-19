
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("iris_dataset.csv")

# Create a folder for graphs
os.makedirs("graphs", exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Bar Chart: Average petal length by species
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="species",
    y="petal length (cm)",
    errorbar=None
)
plt.title("Average Petal Length by Species")
plt.xlabel("Flower Species")
plt.ylabel("Average Petal Length (cm)")
plt.tight_layout()
plt.savefig("graphs/bar_chart.png")
plt.show()

# 2. Histogram: Sepal length distribution
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="sepal length (cm)",
    bins=15,
    kde=True
)
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("graphs/histogram.png")
plt.show()

# 3. Scatter Plot: Petal length vs width
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species",
    s=70
)
plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend(title="Species")
plt.tight_layout()
plt.savefig("graphs/scatter_plot.png")
plt.show()

# 4. Box Plot: Compare petal width
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="species",
    y="petal width (cm)"
)
plt.title("Petal Width Across Species")
plt.xlabel("Flower Species")
plt.ylabel("Petal Width (cm)")
plt.tight_layout()
plt.savefig("graphs/box_plot.png")
plt.show()

# 5. Heatmap: Correlation between measurements
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(8, 6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Between Flower Measurements")
plt.tight_layout()
plt.savefig("graphs/heatmap.png")
plt.show()

print("Data visualization completed successfully!")
print("Five charts saved in the graphs folder.")
