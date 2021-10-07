"""Preprocess videos dataset -> dataset ready to be inputted in application pipelines.

Facade that collects different preprocessing related activities (e.g. face extraction)
and exposes simple interface.

"""
import pathlib
from typing import Generator, Tuple

import cv2 as cv
import numpy as np
from tqdm import tqdm

from .face_extractor import FaceExtractor


def _generate_frame_and_filename_pairs(
    path: pathlib.Path,
) -> Generator[Tuple[np.ndarray, str], None, None]:
    for frame_path in path.iterdir():
        yield cv.imread(str(frame_path)), frame_path.name


def extract_faces_one_by_one(
    face_extractor: FaceExtractor,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
):
    no_frames = sum(1 for _ in input_path.iterdir())
    for frame, file_name in tqdm(_generate_frame_and_filename_pairs(input_path), total=no_frames):
        extracted_face = face_extractor.extract(frame)
        extracted_face_path = output_path.joinpath(file_name)
        cv.imwrite(str(extracted_face_path), extracted_face)
