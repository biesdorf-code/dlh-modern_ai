#!/usr/bin/env python3
""" module for a function to unfreeze the top layers of a base model """


def unfreeze_top_layers(model, n_layers):
    """ unfreezes the last n_layers of model and keeps the rest frozen """
    model.trainable = True  # the parent flag would override its layers
    split = max(len(model.layers) - n_layers, 0)  # safe for 0 or too big
    for layer in model.layers[:split]:
        layer.trainable = False
    for layer in model.layers[split:]:
        layer.trainable = True
