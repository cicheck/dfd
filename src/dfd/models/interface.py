from __future__ import annotations

import abc
import enum
import pathlib
import typing as t


class Prediction(enum.Enum):
    """Represents model prediction."""

    def _generate_next_value_(name, start, count, last_values):
        return name

    REAL = enum.auto()
    FAKE = enum.auto()
    UNCERTAIN = enum.auto()


class ModelInterface(abc.ABC):
    """Height level wrapper around actual models used underneath.

    The goal of exposed interface is to hide implementation details such as what library
    is used to define models. Currently interface operates on paths and handles only data
    stored on disk.

    """

    @abc.abstractmethod
    def train(
        self, train_ds_path: pathlib.Path, validation_ds_path: pathlib.Path
    ) -> t.Dict[str, float]:
        """Train model using given train and validation data.

        Returns:
            metrics computed over validation data

        """

    @abc.abstractmethod
    def test(self, test_ds_path: pathlib.Path) -> t.Dict[str, float]:
        """Evaluate model over provided test data.

        Returns:
            dict, metrics of interests mapped to their values

        """

    @abc.abstractmethod
    def predict(self, sample_path: pathlib.Path) -> t.Dict[pathlib.Path, Prediction]:
        """Make predictions over provided sample of frames."""

    @abc.abstractmethod
    def save(self, path: pathlib.Path):
        """Save model under given path."""

    @classmethod
    @abc.abstractmethod
    def load(cls, path: pathlib.Path) -> ModelInterface:
        """Load model from given path."""

    @abc.abstractmethod
    def get_available_metrics_names(self) -> t.List[str]:
        """Get names of metrics supported by model.

        Each metric value will be returned by train and test functions.

        Returns: names of supported metrics

        """
