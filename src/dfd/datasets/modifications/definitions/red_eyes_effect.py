"""Modification red-eyes effect."""
from typing import Tuple

import cv2 as cv
import dlib
import numpy as np

from dfd.datasets.modifications.specification import ModificationSpecification


def _convert_dlib_shape_to_np_array(dlib_shape) -> np.array:
    """Convert dlib shape into numpy array:

    Args:
        dlib_shape: dliib shape, e.g. output of dlib shape predicator

    Returns:
        numpy array containing points from provided shape

    """
    return np.array([[point.x, point.y] for point in dlib_shape.parts()], dtype="int")


class RedEyesEffectModification(ModificationSpecification):
    """Modification  red-eyes effect."""

    def __init__(self, face_landmarks_detector_path: str, brightness_threshold: int = 50) -> None:
        """Initialize RedEyesEffectModification.

        Args:
            face_landmarks_detector_path: path to dlib trained predictor,
                currently available at http://dlib.net/files/
        """

        self._face_detector = dlib.get_frontal_face_detector()
        self._face_landmarks_detector = dlib.shape_predictor(face_landmarks_detector_path)
        self._brightness_threshold = brightness_threshold

    def __str__(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        """Get specification name.

        Returns:
            The name of specification.

        """
        return f"red_eyes_effect_{self._brightness_threshold}"

    def perform(self, image: np.ndarray) -> np.ndarray:
        """Add red-eyes effect to image.

        Args:
            image: OpenCV image.

        Returns:
            Image with red-eyes effect added, if no face found in image
                original image is returned instead.

        """
        # Convert from BGR color space to YCrCb
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = self._face_detector(gray_image, 1)
        if not faces:
            return image
        face = faces[0]
        shape = self._face_landmarks_detector(gray_image, face)
        landmarks = _convert_dlib_shape_to_np_array(shape)
        # Select eyes landmarks
        left_eye_landmarks = landmarks[36:42]
        right_eye_landmarks = landmarks[42:48]
        modified_image = self._add_red_eye_effect_to_single_eye(image, left_eye_landmarks)
        modified_image = self._add_red_eye_effect_to_single_eye(modified_image, right_eye_landmarks)
        return modified_image

    @staticmethod
    def _create_eye_mask(eye_landmarks: np.ndarray, image_shape: Tuple[int, ...]) -> np.ndarray:
        convex_hull = cv.convexHull(eye_landmarks)
        mask = np.zeros(image_shape, np.uint8)
        mask = cv.drawContours(
            mask, np.array([np.squeeze(convex_hull)]), -1, (255, 255, 255), cv.FILLED
        )
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        return mask

    def _add_red_eye_effect_to_single_eye(
        self, image: np.ndarray, eye_landmarks: np.ndarray
    ) -> np.ndarray:
        eye_mask = self._create_eye_mask(eye_landmarks, image.shape)
        min_channel = np.amin(image, axis=2)
        is_lower_than_threshold = min_channel < self._brightness_threshold
        is_eye_pixel = eye_mask == 255
        is_pupil = is_eye_pixel & is_lower_than_threshold
        addition = 255 - min_channel
        modified_image = image.copy()
        # Modify only red channel
        modified_image[is_eye_pixel & is_pupil, 2] = addition[is_eye_pixel & is_pupil]
        return modified_image
