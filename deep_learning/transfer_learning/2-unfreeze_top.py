#!/usr/bin/env python3
""" module for a function to unfreeze the top layers of a base model """
from tensorflow import keras


def unfreeze_top_layers(model, n_layers):
    """ unfreezes the last n_layers of the base model, rest stays frozen """
    base_model = model
    if len(model.layers) > 1 and isinstance(model.layers[1], keras.Model):
        base_model = model.layers[1]  # full pipeline: base is 2nd layer
    split = max(len(base_model.layers) - n_layers, 0)  # safe for 0 or big
    for layer in base_model.layers[:split]:
        layer.trainable = False
    for layer in base_model.layers[split:]:
        layer.trainable = True
