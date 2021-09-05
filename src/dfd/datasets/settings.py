"""Generator settings."""
import pathlib
from typing import List

import pydantic
import yaml


class ModificationSettings(pydantic.BaseModel):
    """Settings for single modification.

    Args:
        modification_name: name, used to retrieve modification from ModificationRegistry
        share: share of frames on which modification should be applied
        options: modification options, i.e. parameters provided to modification __init__

    """

    name: str
    share: float
    options: dict


class GeneratorSettings(pydantic.BaseModel):
    """Generator settings."""

    modifications: List[ModificationSettings]

    @classmethod
    def from_yaml(cls, yaml_filepath: pathlib.Path) -> "GeneratorSettings":
        """Generate settings from yaml file.

        Args:
            yaml_filepath: Path to file with settings in YAML format.

        Returns:
            Loaded settings.

        """
        with yaml_filepath.open() as yaml_file:
            return cls(**yaml.safe_load(yaml_file))
