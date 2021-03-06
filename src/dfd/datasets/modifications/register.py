"""Modifications register."""
from typing import Dict, Type

from ...exceptions import DfdError
from .definitions import (
    CLAHEModification,
    GammaCorrectionModification,
    GaussianBlurModification,
    GaussianNoiseModification,
    HistogramEqualizationModification,
    MedianFilterModification,
    RedEyesEffectModification,
)
from .interfaces import ModificationInterface

NameToModificationTypeMap = Dict[str, Type[ModificationInterface]]


class ModificationRegister:
    """Define available modifications."""

    def __init__(self, name_to_modification_map: NameToModificationTypeMap) -> None:
        self._name_to_modification_type_map = name_to_modification_map

    @classmethod
    def default(cls) -> "ModificationRegister":
        """Create ModificationRegister with default modifications.

        Returns:
            ModificationRegister with default modifications registered.
        """

        default_modifications = {
            CLAHEModification,
            GammaCorrectionModification,
            HistogramEqualizationModification,
            RedEyesEffectModification,
            GaussianBlurModification,
            MedianFilterModification,
            GaussianNoiseModification,
        }

        return cls(
            {modification.name(): modification for modification in default_modifications},
        )

    def get_modification_class(self, modification_name: str) -> Type[ModificationInterface]:
        """Get registered modification via name.

        Raises:
            DfdError: when targeted modification is not registered

        Returns:
            Type of registered modification.

        """
        try:
            return self._name_to_modification_type_map[modification_name]
        except KeyError:
            raise DfdError(f"Modification {modification_name} is not registered.")
