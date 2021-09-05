"""Perform set modification on provided frames."""
import pathlib

import click
import cv2 as cv
from tqdm import tqdm

from dfd.datasets import GeneratorSettings
from dfd.datasets.frames_generators import ModificationGenerator


@click.command(name="modify")
@click.argument("settings_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("input_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("output_path", type=click.Path(exists=False, path_type=pathlib.Path))
def modify_frames(
    settings_path: pathlib.Path,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
):
    if not settings_path.is_file():
        click.echo("SETTINGS_PATH is not a file")
        return
    if not input_path.is_dir():
        click.echo("INPUT_PATH is not a directory")
        return
    output_path.mkdir(parents=True, exist_ok=True)
    modification_generator = ModificationGenerator(
        settings=GeneratorSettings.from_yaml(settings_path)
    )
    no_frames = sum(1 for _ in input_path.iterdir())
    for modified_frame in tqdm(modification_generator.from_directory(input_path), total=no_frames):
        modified_frame_dir = output_path.joinpath(modified_frame.modification_used)
        modified_frame_dir.mkdir(exist_ok=True)
        modified_frame_path = modified_frame_dir.joinpath(
            modified_frame.original_path.name
        )
        cv.imwrite(str(modified_frame_path), modified_frame.frame)
