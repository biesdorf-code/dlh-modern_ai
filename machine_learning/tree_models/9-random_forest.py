#!/usr/bin/env python3
""" module that builds a random forest classifier """
from sklearn import ensemble


def random_forest(n_estimators, random_state):
    """ Returns a random forest with n_estimators trees """

    model = ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    return model
