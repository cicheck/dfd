"""Generate frames dataset that can be directly fed to pipelines."""
import pathlib
from typing import Generator, Tuple

import click
import cv2 as cv
import numpy as np
from tqdm import tqdm

from dfd.datasets.face_extractor import FaceExtractionModel, FaceExtractor


def _generate_frame_and_filename_pairs(
    path: pathlib.Path,
) -> Generator[Tuple[np.ndarray, str], None, None]:
    for frame_path in path.iterdir():
        yield cv.imread(str(frame_path)), frame_path.name


@click.command(name="generate-fakes")
@click.argument("input_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("output_path", type=click.Path(exists=False, path_type=pathlib.Path))
@click.option(
    "--model-name",
    "--model",
    type=click.Choice(["hog", "cnn"], case_sensitive=False),
    default="hog",
)
def generate_fakes(input_path: pathlib.Path, output_path: pathlib.Path, model_name: str):
    face_extractor = FaceExtractor(FaceExtractionModel(model_name))
    no_fakes = sum(1 for _ in input_path.iterdir())
    output_path.mkdir(parents=True, exist_ok=True)
    for frame, file_name in tqdm(_generate_frame_and_filename_pairs(input_path), total=no_fakes):
        extracted_face = face_extractor.extract(frame)
        extracted_face_path = output_path.joinpath(file_name)
        cv.imwrite(str(extracted_face_path), extracted_face)


@click.command(name="generate-reals")
@click.argument("input_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("output_path", type=click.Path(exists=False, path_type=pathlib.Path))
@click.option("--model-name", "--model", type=click.Choice(["hog", "cnn"], case_sensitive=False))
@click.option(
    "--model-name",
    "--model",
    type=click.Choice(["hog", "cnn"], case_sensitive=False),
    default="hog",
)
def generate_reals():
    pass


# TODO: add option to generate whole
