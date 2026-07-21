#!/usr/bin/env python3
""" module that evaluates classifier performance """
from sklearn import metrics


def evaluate(true_labels, predicted_labels, class_names):
    """ Returns the classification report as a string """

    return metrics.classification_report(true_labels, predicted_labels,
                                         target_names=class_names)
