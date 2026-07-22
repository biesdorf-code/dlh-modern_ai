#!/usr/bin/env python3
""" module for a function to create a Linear model """
from sklearn import linear_model


def Linear_Regression():
    """ creates and returns an untrained model """
    model = linear_model.LinearRegression()  # returns: LinearRegression object
    return model
