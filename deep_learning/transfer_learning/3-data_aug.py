#!/usr/bin/env python3
""" module for a function to build a data augmentation pipeline """
import tensorflow as tf


def build_data_augmentation():
    """ builds a seeded Sequential model of image augmentation layers """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=42),
        tf.keras.layers.RandomRotation(0.15, seed=42),
        tf.keras.layers.RandomZoom(0.15, seed=42),
        tf.keras.layers.RandomContrast(0.1, seed=42),
    ])  # returns: Sequential augmentation model
