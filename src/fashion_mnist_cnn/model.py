"""CNN architecture variants for Fashion-MNIST."""

from dataclasses import dataclass

from tensorflow import keras


CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


@dataclass(frozen=True)
class ModelConfig:
    conv_filters: tuple[int, ...]
    dense_units: int
    dropout_rate: float


MODEL_CONFIGS = {
    "compact": ModelConfig(conv_filters=(16, 32), dense_units=64, dropout_rate=0.20),
    "balanced": ModelConfig(conv_filters=(32, 64), dense_units=128, dropout_rate=0.25),
    "deep": ModelConfig(
        conv_filters=(32, 64, 128),
        dense_units=128,
        dropout_rate=0.30,
    ),
}


def build_classifier(variant: str = "balanced") -> keras.Model:
    """Build and compile one configurable CNN classifier."""
    if variant not in MODEL_CONFIGS:
        supported = ", ".join(MODEL_CONFIGS)
        raise ValueError(f"Unknown model variant {variant!r}. Choose from: {supported}.")

    config = MODEL_CONFIGS[variant]
    layers = [keras.layers.Input(shape=(28, 28, 1), name="image")]

    for block_number, filters in enumerate(config.conv_filters, start=1):
        layers.extend(
            [
                keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=3,
                    padding="same",
                    activation="relu",
                    name=f"conv_{block_number}",
                ),
                keras.layers.MaxPooling2D(pool_size=2, name=f"pool_{block_number}"),
            ]
        )

    layers.extend(
        [
            keras.layers.Flatten(name="flatten"),
            keras.layers.Dense(
                config.dense_units,
                activation="relu",
                name="classifier_hidden",
            ),
            keras.layers.Dropout(config.dropout_rate, name="dropout"),
            keras.layers.Dense(
                len(CLASS_NAMES),
                activation="softmax",
                name="class_probabilities",
            ),
        ]
    )

    model = keras.Sequential(layers, name=f"fashion_mnist_{variant}_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model
