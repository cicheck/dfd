"""Entry point for CLI commands."""

import click

from .predict import predict
from .preprocess import preprocess
from .test import test
from .train import train


@click.group()
def entry_point():
    """Entry point for CLI commands."""


entry_point.add_command(preprocess)
entry_point.add_command(predict)
entry_point.add_command(train)
entry_point.add_command(test)
