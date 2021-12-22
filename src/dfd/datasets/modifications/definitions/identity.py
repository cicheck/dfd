"""Modification Identity."""
import numpy as np

from dfd.datasets.modifications.interfaces import ModificationSpecification


class IdentityModification(ModificationSpecification):
    """Modification Identity."""

    def perform(self, image: np.ndarray) -> np.ndarray:
        """Perform identity modification.

        Returns:
            original image.

        """
        return image

    def __str__(self) -> str:
        return "identity"
