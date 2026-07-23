#!/usr/bin/env python3
""" module for a function to create an SVM model with a chosen kernel """
from sklearn import svm


def get_SVM_model(name, random_state):
    """ creates and returns an untrained SVC model with the given kernel """
    model = svm.SVC(
        kernel=name, random_state=random_state)  # returns: SVC object
    return model
