"""Preprocess CelebDF."""
from pathlib import Path

import click

from dfd.datasets import CelebDFExtractor


@click.group()
def celebdf():
    """Preprocess CelebDF."""


@click.command()
@click.confirmation_option(prompt="Are you sure you want to process whole dataset?")
def whole():
    """Preprocess whole CelebDF.

    If not stopped this command will run until whole dataset is preprocessed.
    """
    click.echo("Not implemented yet.")


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(exists=False, path_type=Path))
@click.option("--lower-bound", "--lower", type=int)
@click.option("--upper-bound", "--upper", type=int)
def reals(input_path: Path, output_path: Path, lower_bound: int, upper_bound: int):
    """Preprocess CelebDF reals."""
    click.echo(
        f"Processing real videos from {lower_bound or 0} up to {upper_bound or 'last video'}"
    )

    # If output path does not exists create it
    output_path.mkdir(exist_ok=True, parents=True)
    preprocessor = CelebDFExtractor(input_path=input_path, output_path=output_path)
    preprocessor.extract_reals_batch(lower_bound=lower_bound, upper_bound=upper_bound)

    click.echo("Done!")


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(exists=False, path_type=Path))
@click.option("--lower-bound", "--lower", type=int)
@click.option("--upper-bound", "--upper", type=int)
def fakes(input_path: Path, output_path: Path, lower_bound: int, upper_bound: int):
    """Preprocess CelebDF fakes."""
    click.echo(
        f"Processing fake videos from {lower_bound or 0} up to {upper_bound or 'last video'}"
    )

    # If output path does not exists create it
    output_path.mkdir(exist_ok=True, parents=True)
    preprocessor = CelebDFExtractor(input_path=input_path, output_path=output_path)
    preprocessor.extract_fakes_batch(lower_bound=lower_bound, upper_bound=upper_bound)

    click.echo("Done!")


# TODO: should be option, not command
celebdf.add_command(whole)
celebdf.add_command(reals)
celebdf.add_command(fakes)
