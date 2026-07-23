#!/usr/bin/env python3
""" module for a function to create a Lasso Regression model """
from sklearn import linear_model


def lasso_regression(random_state):
    """ creates and returns an untrained Lasso model """
    model = linear_model.Lasso(
        random_state=random_state)  # returns: Lasso object
    return model
