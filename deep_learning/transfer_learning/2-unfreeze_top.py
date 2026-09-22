#!/usr/bin/env python3
""" module for a function to unfreeze the top layers of a base model """


def unfreeze_top_layers(model, n_layers):
    """ unfreezes the last n_layers of the base model, rest stays frozen """
    base_model = model
    if len(model.layers) > 1 and hasattr(model.layers[1], "layers"):
        base_model = model.layers[1]  # full pipeline: base is 2nd layer
    split = max(len(base_model.layers) - n_layers, 0)  # safe for 0 or big
    for i, layer in enumerate(base_model.layers):
        layer.trainable = i >= split
