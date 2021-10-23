import pathlib
import sys
import typing as t

import click

from dfd.datasets import FrameExtractor, GeneratorSettings, preprocessor
from dfd.datasets.face_extractor import FaceExtractionModel, FaceExtractor
from dfd.datasets.frames_generators import ModificationGenerator

from .dto import PreprocessDTO, pass_process_dto


@click.command(name="reals")
@click.option(
    "setting_path",
    "--settings",
    type=click.Path(exists=True, path_type=pathlib.Path),
    help=(
        "Path to yaml settings file used to define modifications that will be performed "
        + "on real frames. If no provided default values are used."
    ),
)
@click.argument("storage_path", type=click.Path(exists=False, path_type=pathlib.Path))
@pass_process_dto
def preprocess_reals(
    preprocess_dto: PreprocessDTO,
    setting_path: t.Optional[pathlib.Path],
    storage_path: pathlib.Path,
):
    # TODO: if not settings path provided use some default settings
    if not setting_path.is_file():
        click.echo("Settings path must points to existing file.")
        sys.exit(1)

    storage_path.mkdir(parents=True, exist_ok=True)
    frame_extractor = FrameExtractor()
    face_extractor = FaceExtractor(model=FaceExtractionModel.HOG)
    modification_generator = ModificationGenerator(
        settings=GeneratorSettings.from_yaml(setting_path)
    )
    preprocessor.preprocess_reals(
        frame_extractor=frame_extractor,
        face_extractor=face_extractor,
        modification_generator=modification_generator,
        input_path=preprocess_dto.input_path,
        storage_path=storage_path,
        output_path=preprocess_dto.output_path,
    )
