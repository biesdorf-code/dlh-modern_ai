#!/usr/bin/env python3
""" module that prints the decision rules of a trained tree """
from sklearn import tree


def draw(clf, feature_names, class_names):
    """ Prints the text representation of clf, returns None """

    print(tree.export_text(clf, feature_names=feature_names,
                           class_names=class_names))
