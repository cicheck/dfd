import dataclasses
import pathlib

import click


@dataclasses.dataclass(frozen=True)
class PreprocessDTO:
    """Value object used to pass data from command group to subcommands."""

    input_path: pathlib.Path
    output_path: pathlib.Path


pass_process_dto = click.make_pass_decorator(PreprocessDTO)