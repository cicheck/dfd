"""Preprocess chosen dataset into format that can be directly fed into input pipelines."""

import click

from dfd.cli.preprocess.celeb_df import celebdf


@click.group()
def preprocess():
    """Preprocess chosen dataset into format that can be directly fed into input pipelines."""


preprocess.add_command(celebdf)
