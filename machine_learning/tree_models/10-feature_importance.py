#!/usr/bin/env python3
""" module that computes feature importances from a random forest """
import numpy as np


def feature_importance(rf):
    """ Returns the importances and the indices sorted ascending """

    importances = rf.feature_importances_  # ndarray of floats, sums to 1
    indices = np.argsort(importances)  # ascending, but not the data.

    return importances, indices
