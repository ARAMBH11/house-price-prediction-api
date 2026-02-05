import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def perform_eda(df: pd.DataFrame):
    """
    Perform comprehensive EDA for real estate price prediction.
    """

    sns.set(style="whitegrid")

    print("\n--- Dataset Info ---")
    print(df.info())

    print("\n--- Statistical Summary ---")
    print(df.describe())

    print("\n--- Missing Values ---")
    print(df.isnull().sum().sort_values(ascending=False))

    # Target distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Price"].dropna(), bins=30, kde=True)
    plt.title("Distribution of House Prices")
    plt.show()

    # Outliers in Price
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df["Price"])
    plt.title("Boxplot of House Prices")
    plt.show()

    # Price vs numerical features
    num_features = ["Size", "Bedrooms", "Bathrooms", "Year Built"]
    for col in num_features:
        if col in df.columns:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=df[col], y=df["Price"], alpha=0.3)
            plt.title(f"Price vs {col}")
            plt.show()

    # Price vs categorical features
    cat_features = ["Location", "Type", "Condition"]
    for col in cat_features:
        if col in df.columns:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df[col], y=df["Price"])
            plt.title(f"Price by {col}")
            plt.xticks(rotation=45)
            plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
