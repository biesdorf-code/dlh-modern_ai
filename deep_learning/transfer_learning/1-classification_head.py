#!/usr/bin/env python3
""" module for a function to add a classification head to a base model """
from tensorflow import keras


def add_classification_head(base_model, num_classes):
    """ attaches a Dense(128, relu) + softmax head to a feature extractor """
    x = keras.layers.Dense(128, activation="relu")(base_model.output)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(base_model.input, outputs)  # returns: classifier
