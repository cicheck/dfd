"""Split dataset into training, validation & test sets."""
import pathlib
from typing import Optional

import click
import random

from tqdm import tqdm


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
        click.echo(
            "Ratios [train, validation, test] must adds to one."
        )
        return
    click.echo(f"Splitting with ratios [train: {train}, validation: {validation}, test: {test}].")

    class_directories = [path for path in input_path.iterdir() if path.is_dir()]
    # TODO: loop block to private function
    for class_dir in class_directories:
        all_files = [path for path in class_dir.rglob("*") if path.is_file()]
        no_files = len(all_files)
        # Shuffle files
        random.shuffle(all_files)
        # Split
        validation_lower_bound = int(train * no_files)
        test_lower_bound = int((train + validation) * no_files)
        # Move files
        train_files = all_files[:validation_lower_bound]
        validation_files = all_files[validation_lower_bound:test_lower_bound]
        test_files = all_files[test_lower_bound:]
        click.echo(
            "From class {0} using {1} for train, {2} for validation and {3} for test.".format(
                class_dir.name,
                len(train_files),
                len(validation_files),
                len(test_files),
            )
        )
        # TODO: function instead of 3 loops
        for file in tqdm(train_files, desc="train"):
            path_relative_to_input_dir = file.relative_to(input_path)
            moved_file_path = output_path / "train" / path_relative_to_input_dir
            moved_file_path.parent.mkdir(exist_ok=True, parents=True)
            file.replace(moved_file_path)
        for file in tqdm(validation_files, desc="validation"):
            path_relative_to_input_dir = file.relative_to(input_path)
            moved_file_path = output_path / "validation" / path_relative_to_input_dir
            moved_file_path.parent.mkdir(exist_ok=True, parents=True)
            file.replace(moved_file_path)
        for file in tqdm(test_files, desc="test"):
            path_relative_to_input_dir = file.relative_to(input_path)
            moved_file_path = output_path / "test" / path_relative_to_input_dir
            moved_file_path.parent.mkdir(exist_ok=True, parents=True)
            file.replace(moved_file_path)
