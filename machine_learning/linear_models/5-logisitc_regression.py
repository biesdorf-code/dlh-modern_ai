#!/usr/bin/env python3
""" module for a function to create a Logistic Regression model """
from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """ creates and returns an untrained LogisticRegression model """
    model = linear_model.LogisticRegression(
        random_state=random_state)  # returns: LogisticRegression object
    return model
