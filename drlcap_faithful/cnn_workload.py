"""CNN training workloads of paper Table II: VggNet-16 and GoogleNet trained
on CIFAR-100 (ResNet50 is the paper's substitute for QE on its other GPUs).

Real CIFAR-100 training data (benchmarks/cifar-100/cifar-100-python, the
official python pickle from cs.toronto.edu/~kriz) and the standard
CIFAR-sized architectures:
  vgg16      VGG-16 conv stack (13 conv + 3 FC), BatchNorm, 32x32 input
  googlenet  GoogLeNet / Inception v1 for 32x32 input: the ImageNet stem is
             replaced by one 3x3 conv to 192 channels (usual for 32x32
             CIFAR images); inception 3a-5b keep the original 1x1/3x3/5x5/
             pool branches and channel counts; global average pooling
  resnet50   Keras ResNet50, 32x32 input
Batch 128, SGD momentum 0.9, lr 0.1 (the usual CIFAR-100 recipe). Runs a
fixed number of steps so job length is controlled.

Run by workloads.py as a child process on GPU 1 only, with the CUDA 12
libraries from the base miniconda install on LD_LIBRARY_PATH (the
prasannajeet_env pip CUDA libraries are CUDA 13 builds that the 575 driver
cannot run).

    python cnn_workload.py googlenet --steps 300
"""

import argparse
import os
import pickle

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIFAR = os.path.join(ROOT, "benchmarks", "cifar-100", "cifar-100-python")
BATCH = 128


def load_cifar100():
    with open(os.path.join(CIFAR, "train"), "rb") as f:
        d = pickle.load(f, encoding="latin1")
    x = d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
    mean, std = x.mean(axis=(0, 1, 2)), x.std(axis=(0, 1, 2))
    return (x - mean) / std, np.array(d["fine_labels"], dtype=np.int32)


def conv_bn(x, filters, k=3):
    x = layers.Conv2D(filters, k, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def vgg16():
    inp = layers.Input((32, 32, 3))
    x = inp
    for block in ([64, 64], [128, 128], [256, 256, 256],
                  [512, 512, 512], [512, 512, 512]):
        for f in block:
            x = conv_bn(x, f)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    for _ in range(2):
        x = layers.Dense(4096, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
    return tf.keras.Model(inp, layers.Dense(100)(x), name="vgg16")


def inception(x, c1, c3r, c3, c5r, c5, cp):
    b1 = conv_bn(x, c1, 1)
    b2 = conv_bn(conv_bn(x, c3r, 1), c3)
    b3 = conv_bn(conv_bn(x, c5r, 1), c5, 5)
    b4 = conv_bn(layers.MaxPooling2D(3, 1, padding="same")(x), cp, 1)
    return layers.Concatenate()([b1, b2, b3, b4])


def googlenet():
    inp = layers.Input((32, 32, 3))
    x = conv_bn(inp, 192)
    x = inception(x, 64, 96, 128, 16, 32, 32)       # 3a
    x = inception(x, 128, 128, 192, 32, 96, 64)     # 3b
    x = layers.MaxPooling2D(3, 2, padding="same")(x)
    x = inception(x, 192, 96, 208, 16, 48, 64)      # 4a
    x = inception(x, 160, 112, 224, 24, 64, 64)     # 4b
    x = inception(x, 128, 128, 256, 24, 64, 64)     # 4c
    x = inception(x, 112, 144, 288, 32, 64, 64)     # 4d
    x = inception(x, 256, 160, 320, 32, 128, 128)   # 4e
    x = layers.MaxPooling2D(3, 2, padding="same")(x)
    x = inception(x, 256, 160, 320, 32, 128, 128)   # 5a
    x = inception(x, 384, 192, 384, 48, 128, 128)   # 5b
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    return tf.keras.Model(inp, layers.Dense(100)(x), name="googlenet")


def resnet50():
    return tf.keras.applications.ResNet50(weights=None, input_shape=(32, 32, 3),
                                          classes=100, classifier_activation=None)


MODELS = {"vgg16": vgg16, "googlenet": googlenet, "resnet50": resnet50}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) != 1:
        raise SystemExit(f"expected exactly one visible GPU, got {gpus}")
    tf.config.experimental.set_memory_growth(gpus[0], True)

    x, y = load_cifar100()
    data = (tf.data.Dataset.from_tensor_slices((x, y))
            .shuffle(10000, seed=0).repeat().batch(BATCH)
            .prefetch(tf.data.AUTOTUNE).take(args.steps))
    model = MODELS[args.model]()
    model.compile(optimizer=tf.keras.optimizers.SGD(0.1, momentum=0.9),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(data, epochs=1, verbose=0)


if __name__ == "__main__":
    main()
