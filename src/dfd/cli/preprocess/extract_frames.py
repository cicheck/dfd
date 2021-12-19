import typing as t

import click

from .dto import PreprocessDTO, pass_process_dto
from dfd.datasets import FrameExtractor


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
    """Extract frames from videos contained in given directory.

    Args:
        preprocess_dto:
        lower_bound: Inclusive, lower bound of videos batch that will be processed.
        upper_bound: Exclusive, upper bound of videos batch that will be processed.

    Returns:

    """
    frame_extractor = FrameExtractor()
    frame_extractor.extract_batch(
        input_path=preprocess_dto.input_path,
        output_path=preprocess_dto.output_path,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
