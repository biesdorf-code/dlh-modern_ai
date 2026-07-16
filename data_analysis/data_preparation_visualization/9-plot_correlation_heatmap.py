#!/usr/bin/env python3
""" module for plotting a correlation heatmap """
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    """ plots an annotated correlation heatmap of numeric features """
    plt.figure(figsize=(6, 5))

    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Matrix")

    plt.tight_layout()
    plt.show()

    return None
