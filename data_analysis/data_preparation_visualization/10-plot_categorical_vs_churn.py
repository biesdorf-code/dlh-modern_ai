#!/usr/bin/env python3
""" module for plotting churn rate by categorical feature """
import matplotlib.pyplot as plt


def plot_categorical_vs_churn(df, col):
    """ plots the churn rate (Yes proportion) per category """
    plt.figure(figsize=(12, 8))

    # churn rate = proportion of 'Yes' within each category
    churn_rate = df.groupby(col)['Churn'].apply(
        lambda x: (x == 'Yes').mean())

    plt.bar(churn_rate.index, churn_rate.values)
    plt.title(f"Churn Rate by {col}")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=45)

    plt.show()
