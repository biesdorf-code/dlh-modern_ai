#!/usr/bin/env python3
""" module for a function to compute regre ssion evaluation metrics """
from sklearn import metrics
import numpy as np


def evaluation_metrics_for_regression(y_true, y_pred):
    """ computes and returns mse, rmse, mae, and r2 for regression """
    mse = metrics.mean_squared_error(y_true, y_pred)  # returns: float
    rmse = np.sqrt(mse)  # returns: numpy.float64
    mae = metrics.mean_absolute_error(y_true, y_pred)  # returns: float
    r2 = metrics.r2_score(y_true, y_pred)  # returns: float
    return (mse, rmse, mae, r2)  # returns: tuple of 4 floats
