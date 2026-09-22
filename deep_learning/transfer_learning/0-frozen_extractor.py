#!/usr/bin/env python3
""" module for a function to build a frozen MobileNetV2 feature extractor """
from tensorflow import keras


def build_feature_extractor():
    """ builds a frozen MobileNetV2 base with global average pooling """
    base_model = keras.applications.MobileNetV2(
        weights="imagenet", input_shape=(224, 224, 3), include_top=False)
    base_model.trainable = False  # freeze all pretrained weights

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)  # keep BatchNorm in inference mode
    outputs = keras.layers.GlobalAveragePooling2D()(x)
    return keras.Model(inputs, outputs)  # returns: (None, 1280) features
