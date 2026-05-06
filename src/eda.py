import os
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR


def run_eda(df):
    """
    Основний EDA
    """

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("\n[INFO] Running EDA...")

    plt.figure()
    sns.histplot(df["G3"], bins=20)
    plt.title("Distribution of Final Grade (G3)")
    plt.savefig(FIGURES_DIR / "g3_distribution.png")
    plt.close()

    plt.figure()
    sns.boxplot(x="sex", y="G3", data=df)
    plt.title("G3 by Gender")
    plt.savefig(FIGURES_DIR / "g3_by_gender.png")
    plt.close()

    plt.figure()
    sns.boxplot(x="studytime", y="G3", data=df)
    plt.title("G3 by Study Time")
    plt.savefig(FIGURES_DIR / "g3_by_studytime.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=["int64"])

    corr = numeric_df.corr()
    sns.heatmap(corr, cmap="coolwarm")

    plt.title("Correlation Matrix")
    plt.savefig(FIGURES_DIR / "correlation_matrix.png")
    plt.close()

    print("[INFO] EDA completed. Figures saved.")