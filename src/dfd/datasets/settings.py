"""Generator settings."""
import pathlib
from typing import Dict, List, Union

import pydantic
import yaml

OptionValue = Union[str, int, float]


class ModificationSettings(pydantic.BaseModel):
    """Settings for single modification.

    Args:
        modification_name: name, used to retrieve modification from ModificationRegistry
        share: share of frames on which modification should be applied
        options: modification options, i.e. parameters provided to modification __init__

    """

    modification_name: str
    share: float
    options: Dict[str, OptionValue] = {}


class GeneratorSettings(pydantic.BaseModel):
    """Generator settings."""

    modifications: List[ModificationSettings]

    @classmethod
    def from_yaml(cls, yaml_filepath: pathlib.Path) -> "GeneratorSettings":
        with yaml_filepath.open() as yaml_file:
            return cls(**yaml.safe_load(yaml_file))
