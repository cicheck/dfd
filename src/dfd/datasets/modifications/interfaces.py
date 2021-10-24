"""Interfaces used in modifications package."""
import abc

import numpy as np


class ModificationInterface(abc.ABC):
    """Modification interface."""

    @classmethod
    def name(cls) -> str:
        """Returns modification name."""

        return cls.__name__

    @abc.abstractmethod
    def perform(self, image: np.ndarray) -> np.ndarray:
        """Perform modification on image.

        Args:
            image: OpenCV image.

        Returns:
            Modified image.
        """
