"""Preprocess dataset into format that can be further processed."""

import pathlib

import click

from .dto import PreprocessDTO
from .extract_faces import extract_faces
from .extract_frames import extract_frames
from .preprocess_fakes import preprocess_fakes
from .preprocess_reals import preprocess_reals


@click.group(name="preprocess")
@click.argument("input_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("output_path", type=click.Path(exists=False, path_type=pathlib.Path))
@click.pass_context
def preprocess(ctx, input_path: pathlib.Path, output_path: pathlib.Path):
    output_path.mkdir(parents=True, exist_ok=True)
    ctx.obj = PreprocessDTO(
        input_path=input_path,
        output_path=output_path,
    )


preprocess.add_command(extract_frames)
preprocess.add_command(extract_faces)
preprocess.add_command(preprocess_fakes)
preprocess.add_command(preprocess_reals)
