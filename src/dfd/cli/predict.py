"""Perform prediction for single video or frame."""
import pathlib

import click

from dfd.models import ModelRegistry


@click.command()
@click.option(
    "--model-name",
    type=click.Choice(["meso_net"], case_sensitive=False),
    default="meso_net",
    help="Name of model used.",
)
@click.argument("model_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("data_path", type=click.Path(exists=True, path_type=pathlib.Path))
def predict(model_name: str, model_path: pathlib.Path, data_path: pathlib.Path):
    """Perform prediction on group of videos or frames.

    Args:
        model_name: Name of model which will be trained.
        model_path: Path to model, optional param used to load pre-trained model.
        data_path: Path to data used for predictions

    """
    model_class = ModelRegistry.default().get_model_class(model_name)
    model = model_class.load(model_path)
    frame_path_to_prediction_map = model.predict(sample_path=data_path)
    for frame_path, prediction in frame_path_to_prediction_map.items():
        click.echo(
            "{frame_name}: {prediction}".format(frame_name=frame_path, prediction=prediction)
        )
