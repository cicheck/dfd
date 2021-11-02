import pathlib
import sys
import typing as t

import click

from dfd.datasets import GeneratorSettings, preprocessor
from dfd.datasets.face_extractor import FaceExtractionModel, FaceExtractor
from dfd.datasets.frames_generators import ModificationGenerator

from .dto import PreprocessDTO, pass_process_dto


@click.command(name="modify-frames")
@click.option(
    "setting_path",
    "--settings",
    type=click.Path(exists=True, path_type=pathlib.Path),
    help=(
        "Path to yaml settings file used to define modifications that will be performed "
        + "on real frames. If no provided default values are used."
    ),
)
@pass_process_dto
def modify_frames(
    preprocess_dto: PreprocessDTO,
    setting_path: t.Optional[pathlib.Path],
):
    if setting_path and not setting_path.is_file():
        click.echo("Settings path must points to existing file.")
        sys.exit(1)
    # TODO: Use HOG by default
    face_extractor = FaceExtractor(FaceExtractionModel.CNN, number_of_times_to_upsample=0)
    if setting_path:
        modification_generator_settings = GeneratorSettings.from_yaml(setting_path)
    else:
        modification_generator_settings = GeneratorSettings.default()
    modification_generator = ModificationGenerator(settings=modification_generator_settings)
    preprocessor.modify_frames(
        face_extractor=face_extractor,
        modification_generator=modification_generator,
        input_path=preprocess_dto.input_path,
        output_path=preprocess_dto.output_path,
    )
