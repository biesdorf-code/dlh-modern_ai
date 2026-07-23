#!/usr/bin/env python3
""" module for a function to create a Ridge Regression model """
from sklearn import linear_model


def ridge_regression(random_state):
    """ creates and returns an untrained Ridge model """
    model = linear_model.Ridge(
        random_state=random_state)  # returns: Ridge object
    return model
