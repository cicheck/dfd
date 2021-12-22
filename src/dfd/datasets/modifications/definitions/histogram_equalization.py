"""Modification Histogram Equalization."""

import cv2 as cv
import numpy as np

from dfd.datasets.modifications.interfaces import ModificationSpecification


class HistogramEqualizationModification(ModificationSpecification):
    """Modification Histogram Equalization."""

    def perform(self, image: np.ndarray) -> np.ndarray:
        """Perform Histogram Equalization on image.

        Equalization is done in YCbCr color space,
        after equalization image is converted back to BGR.

        Args:
            image: OpenCV image.

        Returns:
            Image after equalization.

        """
        # Convert from BGR color space to YCrCb
        ycrcb_image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        # Equalize y channel
        ycrcb_image[:, :, 0] = cv.equalizeHist(ycrcb_image[:, :, 0])
        # Convert back to BGR
        return cv.cvtColor(ycrcb_image, cv.COLOR_YCrCb2BGR)

    def __str__(self) -> str:
        return "histogram_equalization"
