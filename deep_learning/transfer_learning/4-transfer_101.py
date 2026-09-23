#!/usr/bin/env python3
""" module for training a Caltech-101 classifier with transfer learning """
import tensorflow as tf
from tensorflow import keras

DATA_DIR = "101_ObjectCategories"
MODEL_PATH = "caltech101_model.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FINETUNE = 15
UNFREEZE_LAYERS = 30
WEIGHTS = "imagenet"


def build_data_augmentation():
    """ builds a seeded Sequential model of image augmentation layers """
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=42),
        keras.layers.RandomRotation(0.1, seed=42),
        keras.layers.RandomZoom(0.1, seed=42),
        keras.layers.RandomContrast(0.1, seed=42),
    ])


def load_datasets():
    """ loads Caltech-101 as preprocessed train / validation datasets """
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="both", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int")
    num_classes = len(train_ds.class_names)  # 101 objects + background

    augment = build_data_augmentation()
    preprocess = keras.applications.mobilenet_v2.preprocess_input
    train_ds = train_ds.map(
        lambda x, y: (preprocess(augment(x, training=True)), y),
        num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(
        lambda x, y: (preprocess(x), y),
        num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, num_classes


def build_model(num_classes):
    """ builds a frozen MobileNetV2 base with a new softmax head """
    base_model = keras.applications.MobileNetV2(
        weights=WEIGHTS, input_shape=IMG_SIZE + (3,), include_top=False)
    base_model.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)  # keep BatchNorm in inference
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs), base_model


def unfreeze_top(base_model, n_layers):
    """ unfreezes the last n_layers of base_model, BatchNorm stays frozen """
    base_model.trainable = True
    split = max(len(base_model.layers) - n_layers, 0)
    for i, layer in enumerate(base_model.layers):
        is_bn = isinstance(layer, keras.layers.BatchNormalization)
        layer.trainable = i >= split and not is_bn


def train_transfer_model():
    """ trains in two phases (head, then fine-tuning) and saves the model """
    train_ds, val_ds, num_classes = load_datasets()
    model, base_model = build_model(num_classes)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2),
    ]

    # Phase 1: train the classification head only
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
              callbacks=callbacks)

    # Phase 2: fine-tune the top layers with a small learning rate
    unfreeze_top(base_model, UNFREEZE_LAYERS)
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE,
              callbacks=callbacks)

    _, val_acc = model.evaluate(val_ds)
    print("Validation accuracy: {:.4f}".format(val_acc))
    model.save(MODEL_PATH)
    return model


if __name__ == "__main__":
    train_transfer_model()
