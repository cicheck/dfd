"""Modification CLAHE (Contrast Limited Adaptive Histogram Equalization)."""

import cv2 as cv
import numpy as np

from dfd.datasets.modifications.interfaces import ModificationInterface


class CLAHEModification(ModificationInterface):
    """Modification CLAHE (Contrast Limited Adaptive Histogram Equalization)"""

    def __init__(self, clip_limit: float, grid_width: int, grid_height: int) -> None:
        """Initialize AdaptiveHistogramEqualizationModification.

        Args:
            clip_limit: limit used to define opencv CLAHE object, as in cv.createCLAHE
            grid_width: tile grid width used to define opencv CLAHE object, as in cv.createCLAHE
            grid_height: tile grid height used to define opencv CLAHE object, as in cv.createCLAHE
        """

        self._clip_limit = clip_limit
        self._title_grid_size = (grid_width, grid_height)

    @classmethod
    def name(cls) -> str:
        """Get name.

        Returns:
            name of modification

        """
        return cls.__name__

    def perform(self, image: np.ndarray) -> np.ndarray:
        """Perform CLAHE on image.

        Equalization is done in YCbCr color space,
        after equalization image is converted back to BGR.

        Args:
            image: OpenCV image.

        Returns:
            Image after equalization.

        """
        # Convert from BGR color space to YCrCb
        ycrcb_image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        # Prepare CLAHE
        clahe = cv.createCLAHE(clipLimit=self._clip_limit, tileGridSize=self._title_grid_size)
        # Equalize y channel
        ycrcb_image[:, :, 0] = clahe.apply(ycrcb_image[:, :, 0])
        # Convert back to BGR
        return cv.cvtColor(ycrcb_image, cv.COLOR_YCrCb2BGR)
