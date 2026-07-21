#!/usr/bin/env python3
""" module that grid searches the best pre-pruning parameters """
from sklearn import model_selection


def prepruning(X, y, clf):
    """ Returns the best pre-pruning hyperparameters found """

    params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 3, 4],
        'min_samples_leaf': [2, 3, 4],
        'min_samples_split': [2, 3, 4]
    }
    grid = model_selection.GridSearchCV(clf, params)
    grid.fit(X, y)
    return grid.best_params_
