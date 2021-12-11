"""Test model on provided data."""
import pathlib

import click

from dfd.models import ModelRegistry

from .utils import echo_metrics


@click.command()
@click.option(
    "--model-name",
    type=click.Choice(["meso_net"], case_sensitive=False),
    default="meso_net",
    help="Name of model used.",
)
@click.argument("model_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("data_path", type=click.Path(exists=True, path_type=pathlib.Path))
def test(model_name: str, model_path: pathlib.Path, data_path: pathlib.Path):
    """Test model on provided data."""
    model_class = ModelRegistry.default().get_model_class(model_name)
    model = model_class.load(model_path)
    metrics_dict = model.test(test_ds_path=data_path)
    echo_metrics(metrics_dict)  # TODO: pretty print
