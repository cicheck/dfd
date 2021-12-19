import click

from dfd.datasets import extract_faces_in_batches, extract_faces_one_by_one
from dfd.datasets.face_extractor import FaceExtractionModel, FaceExtractor

from .dto import PreprocessDTO, pass_process_dto


@click.command(name="extract-faces")
@click.option(
    "--in-batches", is_flag=True, help="Whether to process files in batches or one by one."
)
@click.option(
    "--batch-size",
    type=click.INT,
    default=4,
    help="Batch size, ignored if 'in-batches' flag is off. Should be tuned to fit in GPU.",
)
@click.option(
    "--model-name",
    "--model",
    type=click.Choice(["hog", "cnn"], case_sensitive=False),
    default="hog",
    help=(
        "Model used to find faces if frames are processed one by one, "
        + "if 'in-batches' flag is set cnn is used."
    ),
)
@pass_process_dto
def extract_faces(
    preprocess_dto: PreprocessDTO,
    in_batches: bool,
    model_name: str,
    batch_size: int,
):
    """Extract frames from frames contained in given directory.

    Args:
        preprocess_dto: Object containing input and output path, passed via decorator.
        in_batches: Boolean flag specifying whether to process frames in batches.
        model_name: Name of model used to find faces.
        batch_size: Sie of batch.

    """
    # TODO: Use HOG by default
    face_extractor = FaceExtractor(FaceExtractionModel(model_name), number_of_times_to_upsample=0)
    if not in_batches:
        click.echo("Processing frames one by one...")
        extract_faces_one_by_one(
            face_extractor=face_extractor,
            input_path=preprocess_dto.input_path,
            output_path=preprocess_dto.output_path,
        )
    else:
        click.echo("Processing frames in batches...")
        extract_faces_in_batches(
            face_extractor=face_extractor,
            input_path=preprocess_dto.input_path,
            output_path=preprocess_dto.output_path,
            batch_size=batch_size,
        )
