"""Interfaces used in modifications package."""
import abc

import numpy as np


class ModificationSpecification(abc.ABC):
    """Modification interface."""

    @classmethod
    def class_name(cls) -> str:
        """Get class name.

        Returns:
            Modification class name.

        """
        return cls.__name__

    @abc.abstractmethod
    def perform(self, image: np.ndarray) -> np.ndarray:
        """Perform modification on image.

        Args:
            image: OpenCV image.

        Returns:
            Modified image.
        """
