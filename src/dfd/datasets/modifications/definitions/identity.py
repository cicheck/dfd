"""Modification Identity."""
import numpy as np

from dfd.datasets.modifications.interfaces import ModificationInterface


class IdentityModification(ModificationInterface):
    """Modification Identity."""

    @classmethod
    def name(cls) -> str:
        """Get name.

        Returns:
            name of modification

        """
        return cls.__name__

    def perform(self, image: np.ndarray) -> np.ndarray:
        """Perform identity modification.

        Returns:
            original image.

        """
        return image
