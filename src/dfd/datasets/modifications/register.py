"""Modifications register."""
from typing import Dict, Type

from .definitions import (
    CLAHEModification,
    GammaCorrectionModification,
    HistogramEqualizationModification,
)
from .interfaces import ModificationInterface

NameToModificationTypeMap = Dict[str, Type[ModificationInterface]]


class ModificationRegister:
    """Define available modifications."""

    def __init__(self, name_to_modification_map: NameToModificationTypeMap) -> None:
        self.name_to_modification_type_map = name_to_modification_map

    @classmethod
    def default(cls) -> "ModificationRegister":
        """Create ModificationRegister with default modifications."""

        return cls(
            {
                CLAHEModification.name(): CLAHEModification,
                GammaCorrectionModification.name(): GammaCorrectionModification,
                HistogramEqualizationModification.name(): HistogramEqualizationModification,
            },
        )
