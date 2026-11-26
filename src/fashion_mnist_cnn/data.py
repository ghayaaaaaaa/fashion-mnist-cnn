"""Dataset loading and image preprocessing."""

from typing import NamedTuple

import numpy as np
from tensorflow import keras


class FashionMnistData(NamedTuple):
    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray


def load_fashion_mnist() -> FashionMnistData:
    """Load Keras' Fashion-MNIST data and prepare image tensors for a CNN."""
    (train_images, train_labels), (test_images, test_labels) = (
        keras.datasets.fashion_mnist.load_data()
    )

    return FashionMnistData(
        train_images=np.expand_dims(
            train_images.astype(np.float32) / np.float32(255.0),
            axis=-1,
        ),
        train_labels=train_labels.astype(np.int64),
        test_images=np.expand_dims(
            test_images.astype(np.float32) / np.float32(255.0),
            axis=-1,
        ),
        test_labels=test_labels.astype(np.int64),
    )
