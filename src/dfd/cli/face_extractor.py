"""Generate frames dataset that can be directly fed to pipelines."""
import pathlib

import click

from dfd.datasets import extract_faces_in_batches, extract_faces_one_by_one
from dfd.datasets.face_extractor import FaceExtractionModel, FaceExtractor


@click.command(name="extract-faces")
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
def extract_faces(
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
