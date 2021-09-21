"""Modification Gaussian blur (AKA Gaussian smoothing)."""
from typing import Optional, Tuple

import cv2 as cv
import numpy as np

from dfd.datasets.modifications.interfaces import ModificationInterface


class GaussianBlurModification(ModificationInterface):
    """Modification Gaussian blur (AKA Gaussian smoothing)."""

    def __init__(
        self,
        kernel_size: Optional[Tuple[int, int]] = (5, 5),
        sigma_x: int = 0,
        sigma_y: int = 0,
    ) -> None:
        """Initialize GaussianBlurModification.

        Args:
            kernel_size: Gaussian kernel size used by OpenCV, it needs to be a pair of odd ints,
                or zeros if sigma values should be used to compute kernel automatically.
            sigma_x: Gaussian kernel standard deviation in X direction used by OpenCV.
            sigma_y: Gaussian kernel standard deviation in Y direction used by OpenCV.

        """
        self._kernel_size = kernel_size
        self._sigma_x = sigma_x
        self._sigma_y = sigma_y

    def perform(self, image: np.ndarray) -> np.ndarray:
        """Perform Gaussian blur on provided image.

        Args:
            image: OpenCV image.

        Returns:
            Image after applying Gaussian blur.

        """
        return cv.GaussianBlur(
            image, ksize=self._kernel_size, sigmaX=self._sigma_x, sigmaY=self._sigma_y
        )
