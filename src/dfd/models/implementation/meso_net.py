import pathlib
import typing as t

from tensorflow import keras, metrics
from tensorflow.keras import callbacks, layers, models, optimizers, preprocessing

from ..interface import ModelInterface, Prediction

_IMAGE_SIZE: t.Final = (256, 256)
_MODEL_INPUT_SHAPE: t.Final = (*_IMAGE_SIZE, 3)
_MODEL_THRESHOLD: t.Final = (*_IMAGE_SIZE, 3)


def _build_meso_net_model() -> keras.Sequential:
    model = keras.Sequential()
    model.add(layers.InputLayer(_MODEL_INPUT_SHAPE))
    # First block
    model.add(layers.Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding="same"))
    # Second block
    model.add(layers.Conv2D(8, (5, 5), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding="same"))
    # Third block
    model.add(layers.Conv2D(16, (5, 5), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding="same"))
    # Fourth layer
    model.add(layers.Conv2D(16, (5, 5), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(4, 4), padding="same"))
    # Top
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))

    return model


class MesoNet(ModelInterface):
    """Define Meso-4 model.

    Meso-4 is a relatively shallow DN introduced by Afchar et al..
    ref: "MesoNet: a Compact Facial Video Forgery Detection Network".

    """

    def __init__(self, model: t.Optional[keras.Sequential] = None) -> None:
        self._model: keras.Sequential = model or _build_meso_net_model()
        self._batch_size = 32
        self._metrics = [
            metrics.BinaryAccuracy(),
            metrics.AUC(),
            metrics.Precision(),
            metrics.Recall(),
            metrics.TruePositives(),
            metrics.TrueNegatives(),
            metrics.FalsePositives(),
            metrics.FalseNegatives(),
        ]
        self._callbacks = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
            ),
        ]
        optimizer = optimizers.Adam(
            learning_rate=1e-3,
            epsilon=1e-08,
        )
        self._model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=self._metrics)

    def train(self, train_ds_path: pathlib.Path, validation_ds_path: pathlib.Path) -> None:
        # Load datasets
        train_ds = preprocessing.image_dataset_from_directory(
            train_ds_path,
            batch_size=self._batch_size,
            image_size=_IMAGE_SIZE,
            label_mode="binary",
            class_names=["reals", "fakes"],
        )
        validation_ds = preprocessing.image_dataset_from_directory(
            validation_ds_path,
            batch_size=self._batch_size,
            image_size=_IMAGE_SIZE,
            label_mode="binary",
            class_names=["reals", "fakes"],
        )
        # Calculate reals to fakes ratio
        no_reals = sum(1 for path in train_ds_path.joinpath("reals").rglob("*") if path.is_file())
        no_fakes = sum(1 for path in train_ds_path.joinpath("fakes").rglob("*") if path.is_file())
        reals_to_fake_ratio = no_reals / no_fakes
        self._model.fit(
            train_ds,
            validation_data=validation_ds,
            class_weight={
                # 0 are reals since that's the order used during dataset loading
                0: 1,
                1: reals_to_fake_ratio,
            },
            epochs=10,
            callbacks=self._callbacks,
        )

    def test(self, test_ds_path: pathlib.Path) -> t.Dict[str, float]:
        test_ds = preprocessing.image_dataset_from_directory(
            test_ds_path,
            batch_size=self._batch_size,
            image_size=_IMAGE_SIZE,
            label_mode="binary",
            class_names=["reals", "fakes"],
        )
        return self._model.evaluate(test_ds, return_dict=True)

    def predict(self, sample_path: pathlib.Path) -> t.Dict[pathlib.Path, Prediction]:
        sample_data = preprocessing.image_dataset_from_directory(
            sample_path,
            batch_size=self._batch_size,
            image_size=_IMAGE_SIZE,
            labels=None,
        )
        confidences = self._model.predict(sample_data)
        predictions = [Prediction.from_confidence(confidence).name for confidence in confidences]
        return {path: predictions[idx] for idx, path in enumerate(sample_data.file_paths)}

    def save(self, path: pathlib.Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(path.with_suffix(".h5"))

    def get_available_metrics_names(self) -> t.List[str]:
        return [metric.name for metric in self._metrics]

    @classmethod
    def load(cls, path: pathlib.Path) -> ModelInterface:
        model = models.load_model(path, compile=False)
        return cls(model=model)
