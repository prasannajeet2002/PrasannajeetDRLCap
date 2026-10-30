"""CNN training workloads (paper Table II: VggNet-16 and GoogleNet training;
ResNet50 is the paper's substitute for QE on the RTX 2080 Ti).

Trains a Keras model from scratch on synthetic ImageNet-shaped data for a
fixed number of steps. Data are random so no dataset is needed; the GPU
work (convolutions, cuDNN) is that of real training. Keras has no
GoogLeNet (Inception v1); InceptionV3, its direct successor in the same
family, stands in for it. This is a deviation, stated in the README.

Run by workloads.py as a child process on GPU 1 only, with the CUDA 12
libraries from the base miniconda install on LD_LIBRARY_PATH (the
prasannajeet_env pip CUDA libraries are CUDA 13 builds that the 575 driver
cannot run; see README).

    python cnn_workload.py vgg16 --steps 120
"""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

MODELS = {
    # name: (constructor, input size, batch size)
    "vgg16": (tf.keras.applications.VGG16, 224, 32),
    "googlenet": (tf.keras.applications.InceptionV3, 224, 64),
    "resnet50": (tf.keras.applications.ResNet50, 224, 64),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) != 1:
        raise SystemExit(f"expected exactly one visible GPU, got {gpus}")
    tf.config.experimental.set_memory_growth(gpus[0], True)

    build, size, batch = MODELS[args.model]
    model = build(weights=None, input_shape=(size, size, 3), classes=1000)
    model.compile(optimizer=tf.keras.optimizers.SGD(0.01, momentum=0.9),
                  loss="sparse_categorical_crossentropy")

    rng = np.random.default_rng(0)
    x = rng.standard_normal((batch, size, size, 3), dtype=np.float32)
    y = rng.integers(0, 1000, size=(batch,))
    data = tf.data.Dataset.from_tensors((x, y)).repeat(args.steps)
    model.fit(data, epochs=1, verbose=0)


if __name__ == "__main__":
    main()
