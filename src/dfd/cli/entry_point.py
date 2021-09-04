"""Entry point for CLI commands."""

import click

from dfd.cli.predict import predict
from dfd.cli.extract import extract
from dfd.cli.test import test
from dfd.cli.train import train


@click.group()
def entry_point():
    """Entry point for CLI commands."""


entry_point.add_command(extract)
entry_point.add_command(train)
entry_point.add_command(test)
entry_point.add_command(predict)
