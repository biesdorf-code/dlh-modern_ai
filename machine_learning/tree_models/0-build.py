#!/usr/bin/env python3
""" module that builds a decision tree classifier, clf"""
from sklearn import tree


def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """ Returns a classifier using Gini for impurity """

    model = tree.DecisionTreeClassifier(
        criterion='gini',
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    return model
