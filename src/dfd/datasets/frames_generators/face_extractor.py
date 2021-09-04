"""Generate new frames after performing face extraction on each original."""

import enum
from typing import Generator, Iterable, List, Tuple

import cv2 as cv
import face_recognition
import numpy as np


class FaceExtractionModel(enum.Enum):
    HOG = "hog"
    cnn = "cnn"


# Order is: top, right, bottom, left
FaceLocation = Tuple[int, int, int, int]


class FaceExtractorGenerator:
    """Generate new frames after performing face extraction on each original."""

    def __init__(self, model: FaceExtractionModel, number_of_times_to_upsample: int = 2) -> None:
        self._model_name = model.value
        self._number_of_times_to_upsample = number_of_times_to_upsample

    def generate(self, frames: Iterable[np.ndarray]) -> Generator[np.ndarray, None, None]:
        """Generate frames after face extraction.

        frames: list of opencv images (images in BGR space)

        Yields:
            extracted faces, one per frame if no face was found in frame original frame is yield

        """
        for frame in frames:
            extracted_face = self._extract_face(frame)
            yield extracted_face

    def _extract_face(self, frame) -> np.ndarray:

        frame_in_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        face_locations: List[FaceLocation] = face_recognition.face_locations(
            frame_in_rgb,
            model=self._model_name,
            number_of_times_to_upsample=self._number_of_times_to_upsample,
        )
        # If no face was found return original frame
        if not face_locations:
            return frame
        selected_face_locations = face_locations[0]
        top, right, bottom, left = selected_face_locations
        # Select face
        face = frame[top:bottom, left:right]
        return face
