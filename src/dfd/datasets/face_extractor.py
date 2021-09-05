"""Face extractor."""

import enum
import math
from typing import NamedTuple, Tuple, Sequence, List, Iterable

import cv2 as cv
import face_recognition
import numpy as np

from dfd.consts import MODEL_INPUT_SIZE
from dfd.exceptions import DfdError


class FaceExtractionModel(enum.Enum):
    HOG = "hog"
    cnn = "cnn"


class FaceLocation(NamedTuple):
    """Face location, rectangle."""

    top: int
    right: int
    bottom: int
    left: int

    @classmethod
    def from_tuple(cls, locations_tuple: Tuple[int, int, int, int]) -> "FaceLocation":
        """Convert raw tuple into FaceLocation.

        Args:
            locations_tuple: tuple of face locations, in order: top, right, bottom, left

        Returns:
            FaceLocation instance

        """
        return cls(
            top=locations_tuple[0],
            right=locations_tuple[1],
            bottom=locations_tuple[2],
            left=locations_tuple[3],
        )


class FaceExtractor:
    """Extract face from original frame."""

    def __init__(self, model: FaceExtractionModel, number_of_times_to_upsample: int = 2) -> None:
        self._model_name = model.value
        self._number_of_times_to_upsample = number_of_times_to_upsample

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Extract face from original frame.

        Args:
            frame: OpenCV image. (image in BGR space)

        Returns:
            Single extracted face, if no face was found original image is returned.

        """
        frame_in_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        face_location_raws = face_recognition.face_locations(
            frame_in_rgb,
            model=self._model_name,
            number_of_times_to_upsample=self._number_of_times_to_upsample,
        )
        face_locations = [
            FaceLocation.from_tuple(location_tuple) for location_tuple in face_location_raws
        ]
        # If no face was found return original frame
        if not face_locations:
            return frame
        selected_face_location = face_locations[0]
        face = self._select_face(frame, selected_face_location)
        return face

    def extract_batch(self, frames_batch: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Extract faces from batch of frame.

        Args:
            frames_batch: batch of OpenCV image. (images in BGR space)

        Returns:
            Batch of found faces, one per frame, if no face was found original frame is returned.

        """
        frames_in_rgb = [cv.cvtColor(frame, cv.COLOR_BGR2RGB) for frame in frames_batch]
        faces_batch: List[np.ndarray] = []
        face_locations_batch = face_recognition.batch_face_locations(
            frames_in_rgb,
            batch_size=len(frames_in_rgb),
            number_of_times_to_upsample=self._number_of_times_to_upsample,
        )
        for frame_index, face_location_raws in enumerate(face_locations_batch):
            # If no face was found append original frame
            if not face_location_raws:
                faces_batch.append(frames_batch[frame_index])
                continue
            face_locations = [
                FaceLocation.from_tuple(location_tuple) for location_tuple in face_location_raws
            ]
            selected_face_location = face_locations[0]
            face = self._select_face(frames_batch[frame_index], selected_face_location)
            faces_batch.append(face)
        return faces_batch

    def _select_face(
        self,
        frame: np.ndarray,
        face_location: FaceLocation,
        preferred_size: Tuple[int, int] = MODEL_INPUT_SIZE,
    ) -> np.ndarray:
        top, right, bottom, left = face_location
        pref_width, pref_height = preferred_size
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        # Add margin to bounds
        top, bottom = self._expand_range(top, bottom, pref_height)
        top, bottom = self._adjust_range(top, bottom, frame_height)
        left, right = self._expand_range(left, right, pref_width)
        left, right = self._adjust_range(left, right, frame_width)
        return frame[top:bottom, left:right]

    # TODO: dedicated type / struct for range
    @staticmethod
    def _expand_range(lower_bound: int, upper_bound: int, desired_length: int) -> Tuple[int, int]:
        if lower_bound > upper_bound:
            raise DfdError(f"Incorrect bounds, {lower_bound} > {upper_bound}")
        current_length = upper_bound - lower_bound
        if current_length > desired_length:
            return lower_bound, upper_bound
        margin = math.ceil((desired_length - current_length) / 2)
        lower_bound -= margin
        upper_bound += margin
        return lower_bound, upper_bound

    @staticmethod
    def _adjust_range(
        lower_bound: int, upper_bound: int, max_upper_bound: int, max_lower_bound: int = 0
    ) -> Tuple[int, int]:
        shift = 0
        if upper_bound > max_upper_bound:
            shift += upper_bound - max_upper_bound
        if lower_bound < max_lower_bound:
            shift += max_lower_bound - lower_bound
        lower_bound += shift
        upper_bound += shift
        return lower_bound, upper_bound
