"""Generate frames dataset that can be directly fed to pipelines."""
import pathlib
from typing import Generator, List, Tuple

import click
import cv2 as cv
import numpy as np

from dfd.datasets import extract_faces_in_batches, extract_faces_one_by_one
from dfd.datasets.face_extractor import FaceExtractionModel, FaceExtractor

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


@click.command(name="find-faces")
@click.option("--in-batches", is_flag=True)
@click.option(
    "--batch-size",
    type=click.INT,
    default=4,
    help="Batch size, ignored if 'in-batches' flag is off. Should be tuned to fit in GPU.",
)
@click.argument("input_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("output_path", type=click.Path(exists=False, path_type=pathlib.Path))
@click.option(
    "--model-name",
    "--model",
    type=click.Choice(["hog", "cnn"], case_sensitive=False),
    default="hog",
    help="Model to use if frames are processed one by one, if 'in-batches' flag is set cnn is used.",
)
def find_faces(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    model_name: str,
    in_batches: bool,
    batch_size: int,
):
    face_extractor = FaceExtractor(FaceExtractionModel(model_name), number_of_times_to_upsample=0)
    output_path.mkdir(parents=True, exist_ok=True)
    if not in_batches:
        click.echo("Processing frames one by one...")
        extract_faces_one_by_one(
            face_extractor=face_extractor,
            input_path=input_path,
            output_path=output_path,
        )
    else:
        click.echo("Processing frames in batches...")
        extract_faces_in_batches(
            face_extractor=face_extractor,
            input_path=input_path,
            output_path=output_path,
            batch_size=batch_size,
        )
