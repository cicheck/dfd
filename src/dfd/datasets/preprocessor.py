"""Preprocess videos dataset -> dataset ready to be inputted in application pipelines.

Facade that collects different preprocessing related activities (e.g. face extraction)
and exposes simple interface.

"""
import math
import pathlib
import random
from typing import Generator, List, Optional, Tuple

import cv2 as cv
import numpy as np
from tqdm import tqdm

from dfd.exceptions import DfdError

from .face_extractor import FaceExtractor
from .frame_extractor import FrameExtractor
from .frames_generators import ModificationGenerator

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


def preprocess_fakes(
    frame_extractor: FrameExtractor,
    face_extractor: FaceExtractor,
    input_path: pathlib.Path,
    storage_path: pathlib.Path,
    output_path: pathlib.Path,
    batch_size: Optional[int] = None,
):
    frame_extractor.extract_batch(
        input_path,
        storage_path,
    )
    if not batch_size:
        extract_faces_one_by_one(
            face_extractor,
            input_path=storage_path,
            output_path=output_path,
        )
    else:
        extract_faces_in_batches(
            face_extractor,
            input_path=input_path,
            output_path=output_path,
            batch_size=batch_size,
        )


def preprocess_reals(
    frame_extractor: FrameExtractor,
    face_extractor: FaceExtractor,
    modification_generator: ModificationGenerator,
    input_path: pathlib.Path,
    storage_path: pathlib.Path,
    output_path: pathlib.Path,
):
    frame_extractor.extract_batch(
        input_path,
        storage_path,
    )
    no_frames = sum(1 for _ in input_path.iterdir())
    for modified_frame in tqdm(
        modification_generator.from_directory(storage_path), total=no_frames, desc="real frames"
    ):
        modified_frame_dir = output_path.joinpath(modified_frame.modification_used)
        modified_frame_dir.mkdir(exist_ok=True)
        modified_frame_path = modified_frame_dir.joinpath(modified_frame.original_path.name)
        # Extract faces from modified frames if flag was set
        frame_to_write = modified_frame.frame
        frame_to_write = face_extractor.extract(frame_to_write)
        cv.imwrite(str(modified_frame_path), frame_to_write)


def split(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    train_ds_share: Optional[float] = None,
    validation_ds_share: Optional[float] = None,
    test_ds_share: Optional[float] = None,
):
    """Split dataset into training, validation & test sets.

    Assumptions:
        * input directory contains only subdirectories, each filled with files from particular class

    For input directory with subdirectories:
        * class_a
        * class_b

    Generated structure:
        * train:
            * class_a
            * class_b
        * validation:
            * class_a
            * class_b
        * test:
            * class_a
            * class_b

    Raises:
        DfdError: if specified ds shares are incorrect, see
            ``train_ds_share``, ``validation_ds_share``, ``validation_ds_share``

    Where each class is split according to provided train, validation, test ratios.

    """
    if train_ds_share is None and validation_ds_share is None and test_ds_share is None:
        train_ds_share, validation_ds_share, test_ds_share = 0.6, 0.2, 0.2
    if train_ds_share is None or validation_ds_share is None or test_ds_share is None:
        raise DfdError(
            "If at least one option from dataset share is specified all must be specified."
        )
    if train_ds_share + validation_ds_share + test_ds_share != 1:
        raise DfdError("Dataset shares must adds to one.")

    class_directories = [path for path in input_path.iterdir() if path.is_dir()]
    # TODO: loop block to private function
    for class_dir in class_directories:
        all_files = [path for path in class_dir.rglob("*") if path.is_file()]
        no_files = len(all_files)
        # Shuffle files
        random.shuffle(all_files)
        # Split
        validation_lower_bound = int(train_ds_share * no_files)
        test_lower_bound = int((train_ds_share + validation_ds_share) * no_files)
        # Move files
        train_files = all_files[:validation_lower_bound]
        validation_files = all_files[validation_lower_bound:test_lower_bound]
        test_files = all_files[test_lower_bound:]
        # TODO: function instead of 3 loops
        for file in tqdm(train_files, desc="moving train"):
            path_relative_to_input_dir = file.relative_to(input_path)
            moved_file_path = output_path / "train" / path_relative_to_input_dir
            moved_file_path.parent.mkdir(exist_ok=True, parents=True)
            file.replace(moved_file_path)
        for file in tqdm(validation_files, desc="moving validation"):
            path_relative_to_input_dir = file.relative_to(input_path)
            moved_file_path = output_path / "validation" / path_relative_to_input_dir
            moved_file_path.parent.mkdir(exist_ok=True, parents=True)
            file.replace(moved_file_path)
        for file in tqdm(test_files, desc="moving test"):
            path_relative_to_input_dir = file.relative_to(input_path)
            moved_file_path = output_path / "test" / path_relative_to_input_dir
            moved_file_path.parent.mkdir(exist_ok=True, parents=True)
            file.replace(moved_file_path)
