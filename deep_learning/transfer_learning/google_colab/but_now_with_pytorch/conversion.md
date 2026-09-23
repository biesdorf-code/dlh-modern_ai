# Conversion plan: Keras on TensorFlow → Keras on PyTorch

Source: `../TransferLearning_Caltech101.ipynb` (same logic as `4-transfer_101.py`).
Target: `TransferLearning_Caltech101_pytorch.ipynb` in this folder.

## Goal

Same transfer learning track (MobileNetV2, frozen head training, then
fine-tuning, saved as `caltech101_model.h5`, ≥85% validation accuracy), but
Keras 3 runs on the **PyTorch backend** and the data pipeline is plain
PyTorch. TensorFlow is not imported anywhere.

## What stays the same

- Keras model code: `keras.applications.MobileNetV2(include_top=False)`,
  frozen base called with `training=False`, GAP → Dropout(0.3) → Dense(102).
- `build_model()`, `unfreeze_top()` (last 30 layers, BatchNorm frozen).
- Two phases: Adam(1e-3) for the head, Adam(1e-5) for fine-tuning.
- Callbacks: EarlyStopping (restore best weights), ReduceLROnPlateau.
- `keras.applications.mobilenet_v2.preprocess_input` (scales to [-1, 1]).
- 80/20 split with seed 42, batch size 32, 224×224 images, integer labels
  with `sparse_categorical_crossentropy`.
- Section layout 0–7 and the `train_transfer_model()` function.

## What changes

| Step | TensorFlow version | PyTorch version |
|---|---|---|
| Backend | default (`tensorflow`) | `os.environ["KERAS_BACKEND"] = "torch"` set **before** `import keras` |
| Imports | `import tensorflow as tf`, `from tensorflow import keras` | `import keras`, `import torch`, `torchvision` |
| Loading images | `keras.utils.image_dataset_from_directory` (uses `tf.data`) | `torchvision.datasets.ImageFolder` + `random_split` (seeded) + `DataLoader` |
| Augmentation | Keras layers mapped over `tf.data` | `torchvision.transforms.v2`: **RandomResizedCrop** (the crop/zoom step), horizontal flip, rotation, contrast |
| Preprocessing | `preprocess_input` in `tf.data.map` | same `preprocess_input`, applied in the transform to a channels-last array |
| Batching / prefetch | `tf.data` `.prefetch(AUTOTUNE)` | `DataLoader(num_workers=2, shuffle=True)` |
| Training | `model.fit(tf.data.Dataset)` | `model.fit(DataLoader)` (Keras 3 accepts DataLoaders directly) |
| GPU | TF picks it up | Keras torch backend uses CUDA if available; checked with `torch.cuda.is_available()` |

Notes:
- Keras keeps **channels-last** (`H, W, C`) with the torch backend, so the
  transform returns `(224, 224, 3)` arrays, not PyTorch's usual `(3, 224, 224)`.
- Train and validation use the same split indices but different transforms
  (augmentation on train only), via two `ImageFolder` views + `Subset`.
- The augmentation now includes a random crop (`RandomResizedCrop`,
  scale 0.8–1.0), which plays the role of Keras' `RandomZoom`.

## Curiosity corners (optional, for fun)

Marked `🔍 Curiosity corner` so the main track can be skipped past them.

1. **After loading:** a grid of sample images with their class names.
2. **After augmentation:** 3 images, each shown as the original plus several
   random crops / flips / rotations, to see what the model trains on.
3. **After phase 1:** predictions of the new classification head on
   validation images (true vs predicted class, confidence, green = correct,
   red = wrong). The same helper is called again after fine-tuning.

## How I'll check it before handing over

- Every code cell compiles.
- Run the notebook's code cells locally on a tiny fake dataset with the
  torch backend (no pretrained weights, 1 epoch per phase, plots written
  off-screen) to confirm the pipeline, plots, save and reload work.
- Real accuracy check (≥85%) happens on a Colab GPU run.
