"""Modification Median filter."""

import cv2 as cv
import numpy as np

from dfd.datasets.modifications.specification import ModificationSpecification


class MedianFilterModification(ModificationSpecification):
    """Modification Median filter."""

    def __init__(self, aperture_size: int) -> None:
        """Initialize MedianFilterModification.

        Args:
            aperture_size: Aperture size used by OpenCV, must be odd integer.

        """
        self._aperture_size = aperture_size

    def __str__(self) -> str:
        return "median_filter_{aperture_size}".format(aperture_size=self._aperture_size)

    def perform(self, image: np.ndarray) -> np.ndarray:
        """Perform Median filter modification.

        Args:
            image: OpenCV image.

        Returns:
            Image after Median filter.
        """
        return cv.medianBlur(image, ksize=self._aperture_size)
