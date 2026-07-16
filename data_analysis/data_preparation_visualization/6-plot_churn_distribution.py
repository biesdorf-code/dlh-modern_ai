#!/usr/bin/env python3
""" module for plotting the churn distribution """
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """ plots a bar plot of churn value counts """
    plt.figure(figsize=(12, 8))
    # a series, reindexed in such way that No will appear first.
    counts = df['Churn'].value_counts().reindex(['No', 'Yes'])

    plt.bar(counts.index, counts.values, color=[
            'skyblue', 'salmon'])

    plt.ylabel("Count")
    plt.title("Churn Distribution")
    plt.show()

    return None
