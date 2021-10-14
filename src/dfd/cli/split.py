"""Split dataset into training, validation & test sets."""
import pathlib
from typing import Optional

import click

from dfd.datasets import split as _split


@click.command(name="split")
@click.option("--train", type=click.FLOAT, default=None)
@click.option("--validation", type=click.FLOAT, default=None)
@click.option("--test", type=click.FLOAT, default=None)
@click.argument("input_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("output_path", type=click.Path(exists=False, path_type=pathlib.Path))
def split(
    train: Optional[float],
    validation: Optional[float],
    test: Optional[float],
    input_path: pathlib.Path,
    output_path: pathlib.Path,
):
    """Split dataset into training, validation & test sets.

    Assumptions:
        * input directory contains only subdirectories, each filled with files from particular class

    For input directory with subdirectories:
        * class_a
        * class_b

    Generated structure:
        * train:
            * class_a
            * class_b
        * validation:
            * class_a
            * class_b
        * test:
            * class_a
            * class_b

    Where each class is split according to provided train, validation, test ratios.

    """
    if train is None and validation is None and test is None:
        train, validation, test = 0.6, 0.2, 0.2
    if train is None or validation is None or test is None:
        click.echo(
            "If at least one option from [train, validation, test]"
            + " is specified all must be specified."
        )
        return
    if train + validation + test != 1:
        click.echo("Ratios [train, validation, test] must adds to one.")
        return
    click.echo(f"Splitting with ratios [train: {train}, validation: {validation}, test: {test}].")
    _split(
        input_path=input_path,
        output_path=output_path,
        train_ds_share=train,
        validation_ds_share=validation,
        test_ds_share=test,
    )
