
# CodeAlpha Internship - Task 4
# Product Review Sentiment Analysis

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Create output folder
os.makedirs("graphs", exist_ok=True)

# Load reviews
df = pd.read_csv("reviews.csv")

# Check input data
if "review" not in df.columns:
    raise ValueError("CSV must contain a 'review' column.")

df = df.dropna(subset=["review"]).copy()
df["review"] = df["review"].astype(str).str.strip()
df = df[df["review"] != ""]

if df.empty:
    raise ValueError("No valid reviews found in reviews.csv.")

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Calculate sentiment scores
df["positive_score"] = df["review"].apply(
    lambda text: analyzer.polarity_scores(text)["pos"]
)
df["negative_score"] = df["review"].apply(
    lambda text: analyzer.polarity_scores(text)["neg"]
)
df["neutral_score"] = df["review"].apply(
    lambda text: analyzer.polarity_scores(text)["neu"]
)
df["compound_score"] = df["review"].apply(
    lambda text: analyzer.polarity_scores(text)["compound"]
)

# Classify sentiment using VADER compound score
def classify_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["compound_score"].apply(
    classify_sentiment
)

# Display results
print("\n===== SENTIMENT ANALYSIS RESULTS =====")
print(df[["review", "compound_score", "sentiment"]].to_string(index=False))

# Sentiment counts and percentages
counts = df["sentiment"].value_counts().reindex(
    ["Positive", "Neutral", "Negative"],
    fill_value=0
)

percentages = (counts / len(df) * 100).round(2)

summary = pd.DataFrame({
    "Count": counts,
    "Percentage": percentages
})

print("\n===== SENTIMENT SUMMARY =====")
print(summary)

# Save results
df.to_csv("sentiment_results.csv", index=False)

# Chart 1: Sentiment distribution
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="sentiment",
    order=["Positive", "Neutral", "Negative"]
)
plt.title("Customer Review Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("graphs/sentiment_distribution.png")
plt.close()

# Chart 2: Compound sentiment scores
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="compound_score",
    bins=10,
    kde=True
)
plt.title("Distribution of Compound Sentiment Scores")
plt.xlabel("Compound Score (-1 to +1)")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("graphs/sentiment_scores.png")
plt.close()

# Overall interpretation
positive_pct = percentages["Positive"]
negative_pct = percentages["Negative"]

print("\n===== KEY INSIGHTS =====")
print(f"Total reviews analyzed: {len(df)}")
print(f"Positive reviews: {positive_pct}%")
print(f"Neutral reviews: {percentages['Neutral']}%")
print(f"Negative reviews: {negative_pct}%")

if positive_pct > negative_pct:
    print("Overall finding: More reviews are positive than negative.")
elif negative_pct > positive_pct:
    print("Overall finding: More reviews are negative than positive.")
else:
    print("Overall finding: Positive and negative review percentages are equal.")

print("\nAnalysis completed successfully!")
print("Results saved to sentiment_results.csv")
print("Charts saved in the graphs folder.")
