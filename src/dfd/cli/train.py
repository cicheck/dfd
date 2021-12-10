"""Train model on provided data."""
import pathlib
import typing as t

import click

from dfd.models import ModelRegistry


@click.command()
@click.option(
    "--model-name",
    type=click.Choice(["meso_net"], case_sensitive=False),
    default="meso_net",
    help="Name of model used.",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    help="Optional path used to load already pre-trained model.",
)
@click.option(
    "--output-path",
    type=click.Path(exists=False, path_type=pathlib.Path),
    help="Optional path used to save trained model model.",
)
@click.argument("train_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("validation_path", type=click.Path(exists=True, path_type=pathlib.Path))
def train(
    model_name: str,
    model_path: t.Optional[pathlib.Path],
    output_path: t.Optional[pathlib.Path],
    train_path: pathlib.Path,
    validation_path: pathlib.Path,
):
    """Train model on provided data.

    Args:
        model_name: Name of model which will be trained.
        model_path: Path to model, optional param used to load pre-trained model.
        output_path: Path that will be used to store trained model.
        train_path: Path to train dataset.
        validation_path: Path to validation dataset.

    """
    model_class = ModelRegistry.default().get_model_class(model_name)

    if model_path:
        model = model_class.load(model_path)
    else:
        model = model_class()  # TODO: method default instead of __init__ that can be overwrite
    model.train(train_ds_path=train_path, validation_ds_path=validation_path)
    if output_path:
        model.save(output_path)
