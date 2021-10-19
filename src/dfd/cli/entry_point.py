"""Entry point for CLI commands."""

import click

from .face_extractor import extract_faces
from .frames_extractor import extract_frames
from .modification_generator import modify_frames
from .predict import predict
from .preprocessor import preprocess
from .split import split
from .test import test
from .train import train


@click.group()
def entry_point():
    """Entry point for CLI commands."""


entry_point.add_command(preprocess)
entry_point.add_command(extract_frames)
entry_point.add_command(extract_faces)
entry_point.add_command(modify_frames)
entry_point.add_command(predict)
entry_point.add_command(split)
entry_point.add_command(train)
entry_point.add_command(test)
