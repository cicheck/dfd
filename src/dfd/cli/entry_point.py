"""Entry point for CLI commands."""

import click

from .frames_extractor import extract_frames
from .generate import generate_fakes, generate_reals
from .predict import predict
from .test import test
from .train import train


@click.group()
def entry_point():
    """Entry point for CLI commands."""


entry_point.add_command(extract_frames)
entry_point.add_command(generate_fakes)
entry_point.add_command(generate_reals)
entry_point.add_command(train)
entry_point.add_command(test)
entry_point.add_command(predict)
