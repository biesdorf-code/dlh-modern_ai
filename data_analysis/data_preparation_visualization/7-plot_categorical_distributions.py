#!/usr/bin/env python3
""" module for plotting categorical feature distributions """
import matplotlib.pyplot as plt


def plot_categorical_distributions(df, columns_to_plot=None):
    """ plots bar plots for each categorical feature in a grid """
    if columns_to_plot is None:
        df = df.drop("Churn", axis=1)
        df = df.select_dtypes(include="object")
        columns_to_plot = df.columns.to_list()  # to_list() returns a list
    else:
        if isinstance(columns_to_plot, str):
            columns_to_plot = [columns_to_plot]  # convert to list
    num_plots = len(columns_to_plot)
    n_cols, n_rows = 3, (num_plots+2)//3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))

    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for ax, col in zip(axes, columns_to_plot):
        counts = df[col].value_counts()  # value_counts() returns a Series
        ax.bar(counts.index, counts.values)
        ax.set_title(col)
        ax.tick_params(axis='x', labelrotation=45)

    for ax in axes[num_plots:]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("Task_7.png")
    plt.show()

    return None
