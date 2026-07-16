#!/usr/bin/env python3
""" module for visualizing missing data """
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """ scatter plot where each missing value is a small pipe"""
    plt.figure(figsize=(12, 8))

    for i, col in enumerate(df.columns):
        missing = df.index[df[col].isnull()]
        plt.scatter(missing, [i] * len(missing), marker='|')

    plt.yticks(range(len(df.columns)), df.columns)
    plt.title("Missingness Plot")

    plt.tight_layout()
    plt.show()
