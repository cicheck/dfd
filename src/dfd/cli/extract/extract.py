"""Extract frames from videos."""

import click

from dfd.cli.extract.celeb_df import celebdf


@click.group()
def extract():
    """Extract frames from videos."""

# TODO: required argument dataset, or optional dataset & be default extract from single video
extract.add_command(celebdf)
