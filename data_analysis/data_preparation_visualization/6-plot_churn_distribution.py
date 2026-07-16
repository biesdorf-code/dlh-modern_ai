#!/usr/bin/env python3
""" module for plotting the churn distribution """
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """ plots a bar plot of churn value counts """
    plt.figure(figsize=(12, 8))

    counts = df['Churn'].value_counts()
    plt.bar(counts.index, counts.values, color=['skyblue', 'salmon'])
    plt.title("Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
