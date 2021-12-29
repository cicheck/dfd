"""Modification Identity."""
import numpy as np

from dfd.datasets.modifications.specification import ModificationSpecification


class IdentityModification(ModificationSpecification):
    """Modification Identity."""

    def perform(self, image: np.ndarray) -> np.ndarray:
        """Perform identity modification.

        Returns:
            original image.

        """
        return image

    @property
    def name(self) -> str:
        """Get specification name.

        Returns:
            The name of specification.

        """
        return "identity"

    def __str__(self) -> str:
        return self.name
