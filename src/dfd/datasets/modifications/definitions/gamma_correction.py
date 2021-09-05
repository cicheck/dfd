"""Modification Gamma Correction."""
import cv2 as cv
import numpy as np

from dfd.datasets.modifications.interfaces import ModificationInterface


class GammaCorrectionModification(ModificationInterface):
    """Modification Gamma Correction."""

    def __init__(self, gamma_value: float) -> None:
        """Initialize GammaCorrectionModification.

        Args:
            gamma_value: gamma value, must be positive number

        """
        self._gamma_value = gamma_value

    @classmethod
    def name(cls) -> str:
        """Get name.

        Returns:
            name of modification

        """
        return cls.__name__

    def perform(self, image: np.ndarray) -> np.ndarray:
        """Perform gamma correction on provided image.

        Args:
            image: OpenCV image.

        Returns:
            Image after gamma correction.

        """
        rgb_max_value = 255
        look_up_table = np.array(
            [
                int(((i / rgb_max_value) ** (1.0 / self._gamma_value)) * rgb_max_value)
                for i in np.arange(0, 256)
            ]
        ).astype("uint8")
        # apply gamma correction using lookup table
        return cv.LUT(image, look_up_table)

    def __str__(self) -> str:
        return f"gamma_correction_{self._gamma_value}"
