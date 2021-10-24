"""Modification Gaussian blur (AKA Gaussian smoothing)."""

import cv2 as cv
import numpy as np

from dfd.datasets.modifications.interfaces import ModificationInterface


class GaussianBlurModification(ModificationInterface):
    """Modification Gaussian blur (AKA Gaussian smoothing)."""

    def __init__(
        self,
        kernel_width: int = 0,
        kernel_height: int = 0,
        sigma_x: int = 0,
        sigma_y: int = 0,
    ) -> None:
        """Initialize GaussianBlurModification.

        Args:
            kernel_width: Width of kernel used by OpenCV, should be zero or odd int.
            kernel_height: Height of kernel used by OpenCV, should be zero or odd int.
            sigma_x: Gaussian kernel standard deviation in X direction used by OpenCV.
            sigma_y: Gaussian kernel standard deviation in Y direction used by OpenCV.

        """
        self._kernel_size = (kernel_width, kernel_height)
        self._sigma_x = sigma_x
        self._sigma_y = sigma_y

    def __str__(self) -> str:
        width, height = self._kernel_size
        return f"gaussian_blur{width}_{height}_{self._sigma_x}_{self._sigma_y}"

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
