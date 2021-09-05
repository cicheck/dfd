"""Extract frames from videos."""

import click

from dfd.cli.frames_extractor.celeb_df import celebdf


@click.group(name="extract-frames")
def extract_frames():
    """Extract frames from videos."""

# TODO: required argument dataset, or optional dataset & be default frames_extractor from single video
extract_frames.add_command(celebdf)
