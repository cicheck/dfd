import pathlib
import sys
import typing as t

import click

from dfd.datasets import FrameExtractor, GeneratorSettings, preprocessor
from dfd.datasets.face_extractor import FaceExtractionModel, FaceExtractor
from dfd.datasets.frames_generators import ModificationGenerator

from .dto import PreprocessDTO, pass_process_dto


@click.command(name="directory")
@click.option(
    "setting_path",
    "--settings",
    type=click.Path(exists=True, path_type=pathlib.Path),
    help=(
        "Path to yaml settings file used to define modifications that will be performed "
        + "on real frames. If no provided default values are used."
    ),
)
@click.option(
    "--batch-size",
    type=click.INT,
    help=(
        "Batch size provided for model used to extract faces."
        + "If not specified frames are processed one by one."
    ),
)
@click.option(
    "--model-name",
    "--model",
    type=click.Choice(["hog", "cnn"], case_sensitive=False),
    default="cnn",
    help=(
        "Model used to find faces if frames are processed one by one, "
        + "if 'in-batches' flag is set cnn is used."
    ),
)
@click.argument("storage_path", type=click.Path(exists=False, path_type=pathlib.Path))
@pass_process_dto
def preprocess_directory(
    preprocess_dto: PreprocessDTO,
    setting_path: t.Optional[pathlib.Path],
    batch_size: t.Optional[int],
    model_name: str,
    storage_path: pathlib.Path,
):
    """Preprocess directory containing fake and real videos.

    Args:
        preprocess_dto: Object containing input and output path, passed via decorator.
        setting_path: Path to settings used to define modifications used.
        batch_size: Size of batch that will be used during face finding.
        model_name: Name of model used to find faces.
        storage_path: Path that will be used to store frames extracted from videos.

    """
    if setting_path and not setting_path.is_file():
        click.echo("Settings path must points to existing file.")
        sys.exit(1)

    storage_path.mkdir(parents=True, exist_ok=True)
    frame_extractor = FrameExtractor()
    # TODO: use HOG by default
    face_extraction_model = FaceExtractionModel(model_name)
    face_extractor = FaceExtractor(face_extraction_model, number_of_times_to_upsample=0)
    # HOG cannot process frames in batches
    # TODO: verify
    if face_extraction_model == FaceExtractionModel.HOG:
        batch_size = None
    if setting_path:
        modification_generator_settings = GeneratorSettings.from_yaml(setting_path)
    else:
        modification_generator_settings = GeneratorSettings.default()
    modification_generator = ModificationGenerator(settings=modification_generator_settings)
    preprocessor.preprocess_single_directory(
        frame_extractor=frame_extractor,
        face_extractor=face_extractor,
        modification_generator=modification_generator,
        input_path=preprocess_dto.input_path,
        storage_path=storage_path,
        output_path=preprocess_dto.output_path,
        batch_size=batch_size,
    )
