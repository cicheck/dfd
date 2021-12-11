"""Model stubs."""

import pathlib
import typing as t

from dfd.models import ModelInterface
from dfd.models.interface import Prediction


class ModelStub(ModelInterface):
    def train(
        self, train_ds_path: pathlib.Path, validation_ds_path: pathlib.Path
    ) -> t.Dict[str, float]:
        return {"metric": 0}

    def test(self, test_ds_path: pathlib.Path) -> t.Dict[str, float]:
        return {"metric": 0}

    def predict(self, sample_path: pathlib.Path) -> t.Dict[pathlib.Path, Prediction]:
        return {
            pathlib.Path("real"): Prediction.REAL,
            pathlib.Path("fake"): Prediction.FAKE,
            pathlib.Path("uncertain"): Prediction.UNCERTAIN,
        }

    def save(self, path: pathlib.Path):
        return

    @classmethod
    def load(cls, path: pathlib.Path) -> ModelInterface:
        return cls()

    def get_available_metrics_names(self) -> t.List[str]:
        return ["metric"]
