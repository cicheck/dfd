import pathlib
import typing as t

import click

from dfd.datasets import FrameExtractor, preprocessor
from dfd.datasets.face_extractor import FaceExtractionModel, FaceExtractor

from .dto import PreprocessDTO, pass_process_dto


@click.command(name="fakes")
@click.option(
    "--batch-size",
    type=click.INT,
    help=(
        "Batch size provided for model used to extract faces."
        + "If not specified frames are processed one by one."
    ),
)
@click.argument("storage_path", type=click.Path(exists=False, path_type=pathlib.Path))
@pass_process_dto
def preprocess_fakes(
    preprocess_dto: PreprocessDTO,
    batch_size: t.Optional[int],
    storage_path: pathlib.Path,
):
    storage_path.mkdir(parents=True, exist_ok=True)
    frame_extractor = FrameExtractor()
    # If batch size is set use CNN model
    if batch_size:
        face_extractor_model_type = FaceExtractionModel.CNN
    else:
        face_extractor_model_type = FaceExtractionModel.HOG
    face_extractor = FaceExtractor(model=face_extractor_model_type)
    preprocessor.preprocess_fakes(
        frame_extractor=frame_extractor,
        face_extractor=face_extractor,
        input_path=preprocess_dto.input_path,
        storage_path=storage_path,
        output_path=preprocess_dto.output_path,
        batch_size=batch_size,
    )
