"""Preprocess dataset into format that can be further processed."""

import dataclasses
import pathlib
import typing as t

import click

from dfd.datasets import FrameExtractor


@dataclasses.dataclass(frozen=True)
class PreprocessDTO:
    """Value object used to pass data from command group to subcommands."""

    input_path: pathlib.Path
    output_path: pathlib.Path


pass_process_dto = click.make_pass_decorator(PreprocessDTO)


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


@click.command(name="extract-frames")
@click.option(
    "--lower-bound",
    "--lower",
    type=int,
    help="specify lower index of videos range that will be processed, defaults to 0",
)
@click.option(
    "--upper-bound",
    "--upper",
    type=int,
    help="specify upper index of videos range that will be processed, defaults to last available",
)
@pass_process_dto
def extract_frames(
    preprocess_dto: PreprocessDTO, lower_bound: t.Optional[int], upper_bound: t.Optional[int]
):
    frame_extractor = FrameExtractor()
    frame_extractor.extract_batch(
        input_path=preprocess_dto.input_path,
        output_path=preprocess_dto.output_path,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


preprocess.add_command(extract_frames)
