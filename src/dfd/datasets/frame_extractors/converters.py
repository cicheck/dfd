"""Converters used in frame extractors package."""

import cv2
import numpy as np


def convert_video_to_frames(filepath: str) -> list[np.ndarray]:
    """Split video into frames.

    Args:
        filepath: path to video

    Returns:
        list of frames

    """

    frames: list[np.ndarray] = []
    capture = cv2.VideoCapture(filepath)

    while True:
        success, frame = capture.read()
        if not success:
            break
        frames.append(frame)

    return frames
