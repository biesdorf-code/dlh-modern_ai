#!/usr/bin/env python3
""" module that generates predictions from a trained tree """


def generate_predictions(clf, X):
    """ Returns the predicted class labels for X """

    return clf.predict(X)
