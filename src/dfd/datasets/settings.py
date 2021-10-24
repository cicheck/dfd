"""Generator settings."""
import pathlib
from typing import List

import pydantic
import yaml

# TODO: include model in project
_FACE_DETECTOR_PATH = "/media/cicheck/Extreme Pro/models/shape_predictor_68_face_landmarks.dat"


class ModificationSettings(pydantic.BaseModel):
    """Settings for single modification.

    Args:
        modification_name: name, used to retrieve modification from ModificationRegistry
        share: share of frames on which modification should be applied
        options: modification options, i.e. parameters provided to modification __init__

    """

    name: str
    share: float
    options: dict = {}


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

    @classmethod
    def default(cls) -> "GeneratorSettings":
        """Generate settings using default values.

        Returns:
            Default settings.

        """
        return cls(
            modifications=[
                ModificationSettings(
                    name="RedEyesEffectModification",
                    share="0.125",
                    options={
                        "brightness_threshold": 50,
                        "face_landmarks_detector_path": _FACE_DETECTOR_PATH,
                    },
                ),
                ModificationSettings(
                    name="CLAHEModification",
                    share="0.125",
                    options={
                        "clip_limit": 2.0,
                        "grid_width": 8,
                        "grid_height": 8,
                    },
                ),
                ModificationSettings(
                    name="HistogramEqualizationModification",
                    share="0.125",
                    options={},
                ),
                ModificationSettings(
                    name="GammaCorrectionModification",
                    share="0.0625",
                    options={"gamma_value": 0.75},
                ),
                ModificationSettings(
                    name="GammaCorrectionModification",
                    share="0.0625",
                    options={"gamma_value": 1.25},
                ),
            ]
        )
