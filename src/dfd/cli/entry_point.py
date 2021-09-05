"""Entry point for CLI commands."""

import click

from .frames_extractor import extract_frames
from .face_finder import find_faces
from .modification_generator import modify_frames
from .predict import predict
from .test import test
from .train import train


@click.group()
def entry_point():
    """Entry point for CLI commands."""


entry_point.add_command(extract_frames)
entry_point.add_command(find_faces)
entry_point.add_command(modify_frames)
entry_point.add_command(train)
entry_point.add_command(test)
entry_point.add_command(predict)
