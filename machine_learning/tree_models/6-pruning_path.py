#!/usr/bin/env python3
""" module that retrieves the cost-complexity pruning path """


def get_pruning_path(clf, X, y):
    """ Returns the ccp_alphas and impurities of the pruning path """

    path = clf.cost_complexity_pruning_path(X, y)
    return path.ccp_alphas, path.impurities
