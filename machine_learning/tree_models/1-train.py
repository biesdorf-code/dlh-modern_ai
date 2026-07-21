#!/usr/bin/env python3
""" module that trains a tree-based classifier """


def train_tree(clf, X, y):
    """ Fits clf on features X and labels y, returns None """

    clf.fit(X, y)
