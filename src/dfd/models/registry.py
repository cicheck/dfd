from __future__ import annotations

import typing as t
from types import MappingProxyType

from . import implementation

from ..exceptions import DfdError
from .interface import ModelInterface

NameToModelClassMap = MappingProxyType[str, t.Type[ModelInterface]]


class ModelRegistry:
    """Register available models."""

    def __init__(self, name_to_model_class_map: NameToModelClassMap) -> None:
        self._name_to_model_class_map = name_to_model_class_map

    @classmethod
    def default(cls) -> ModelRegistry:
        """Create `ModelRegistry` with default models.

        Returns:
            `ModelRegistry` instance with default models registered.

        """

        return cls(
            MappingProxyType({"meso_net": implementation.MesoNet}),
        )

    def get_model_class(self, model_name: str) -> t.Type[ModelInterface]:
        """Get registered model via name.

        Raises:
            DfdError: when targeted modification is not registered

        Returns:
            Class of registered model.

        """
        try:
            return self._name_to_model_class_map[model_name]
        except KeyError:
            raise DfdError(f"Model {model_name} is not registered.")
