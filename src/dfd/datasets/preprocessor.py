"""Preprocess videos dataset -> dataset ready to be inputted in application pipelines.

Facade that collects different preprocessing related activities (e.g. face extraction)
and exposes simple interface.

"""
import math
import pathlib
from typing import Generator, List, Tuple

import cv2 as cv
import numpy as np
from tqdm import tqdm

from .face_extractor import FaceExtractor

FrameAndNamePair = Tuple[np.ndarray, str]


def _generate_frame_and_filename_pairs(
    path: pathlib.Path,
) -> Generator[Tuple[np.ndarray, str], None, None]:
    for frame_path in path.iterdir():
        yield cv.imread(str(frame_path)), frame_path.name


def _generate_frame_and_filename_batches(
    path: pathlib.Path,
    batch_size: int = 64,
) -> Generator[List[FrameAndNamePair], None, None]:
    batch: List[FrameAndNamePair] = []
    for frame_path in path.iterdir():
        frame_and_name_pair = (cv.imread(str(frame_path)), frame_path.name)
        # Frame has different shape than previous ones (i.e. is from different video)
        # TODO: ugly use named tuple instead of [0][0]
        if len(batch) > 0 and batch[0][0].shape != frame_and_name_pair[0].shape:
            yield batch
            batch = [frame_and_name_pair]
            continue

        batch.append(frame_and_name_pair)
        # Max batch size achieved
        if len(batch) == batch_size:
            yield batch
            batch = []
    # Leftovers
    if len(batch) > 0:
        yield batch


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


def extract_faces_in_batches(
    face_extractor: FaceExtractor,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    batch_size: int,
):
    no_frames = sum(1 for _ in input_path.iterdir())
    no_batches = math.ceil(no_frames / batch_size)
    for batch in tqdm(
        _generate_frame_and_filename_batches(input_path, batch_size=batch_size),
        total=no_batches,
    ):
        frames_batch, names_batch = zip(*batch)
        face_batch = face_extractor.extract_batch(frames_batch)
        for frame_index, face in enumerate(face_batch):
            face_path = output_path.joinpath(names_batch[frame_index])
            cv.imwrite(str(face_path), face)
