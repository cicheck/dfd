"""Generator settings."""
import pathlib
from typing import List

import pydantic
import yaml

# TODO: include model in project
from dfd import assets


class ModificationSettings(pydantic.BaseModel):
    """Settings for single modification.

    Args:
        modification_name: name, used to retrieve modification from ModificationRegistry
        options: modification options, i.e. parameters provided to modification `__init__`

    """

    name: str
    options: dict = {}


class ModificationsChainSettings(pydantic.BaseModel):
    """Settings for single chain of modification applied in order.

    Args:
        share: Share of frames on which modifications chain will be applied.
        modifications: List of modifications that will be applied on sectioned frames.
            Modification on top will be applied first, modification on bootom last.

    """

    share: float
    modifications: List[ModificationSettings] = []


class GeneratorSettings(pydantic.BaseModel):
    """Generator settings."""

    modifications_chains: List[ModificationsChainSettings]

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
            modifications_chains=[
                ModificationsChainSettings(
                    share="0.125",
                    modifications=[
                        ModificationSettings(
                            name="RedEyesEffectModification",
                            options={
                                "brightness_threshold": 50,
                                "face_landmarks_detector_path": str(
                                    assets.FACE_LANDMARKS_MODEL_PATH
                                ),
                            },
                        ),
                    ],
                ),
                ModificationsChainSettings(
                    share="0.125",
                    modifications=[
                        ModificationSettings(
                            name="CLAHEModification",
                            options={
                                "clip_limit": 2.0,
                                "grid_width": 8,
                                "grid_height": 8,
                            },
                        ),
                    ],
                ),
                ModificationsChainSettings(
                    share="0.125",
                    modifications=[
                        ModificationSettings(
                            name="HistogramEqualizationModification",
                            options={},
                        ),
                    ],
                ),
                ModificationsChainSettings(
                    share="0.0625",
                    modifications=[
                        ModificationSettings(
                            name="GammaCorrectionModification",
                            options={"gamma_value": 0.75},
                        ),
                    ],
                ),
                ModificationsChainSettings(
                    share="0.0625",
                    modifications=[
                        ModificationSettings(
                            name="GammaCorrectionModification",
                            options={"gamma_value": 1.25},
                        ),
                    ],
                ),
            ]
        )
