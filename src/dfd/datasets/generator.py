"""Generate new frames after performing set non malicious modifications on original frames."""
import functools
import pathlib
from typing import Generator, List, NamedTuple, Optional

import cv2 as cv
import numpy as np

from ..exceptions import DfdError
from .modifications.definitions import IdentityModification
from .modifications.interfaces import ModificationInterface
from .modifications.register import ModificationRegister
from .settings import GeneratorSettings


class ModificationShare(NamedTuple):
    """Share of frames on which modification will be performed."""

    modification: ModificationInterface
    share: float


class ModificationRange(NamedTuple):
    """Range of frames on which set modification will be performed.

    Args:
        modification: Modification to be performed
        lower_bound: lower range bound, inclusive
        upper_bound: upper range bound, inclusive

    """

    modification: ModificationInterface
    lower_bound: int
    upper_bound: int


class ModifiedFrame(NamedTuple):
    """Modified frame."""

    modification_used: str
    frame: np.ndarray


class FramesGenerator:
    """Generate new frames after performing set non malicious modifications on original frames."""

    def __init__(
        self,
        settings: GeneratorSettings,
        register: Optional[ModificationRegister] = None,
    ) -> None:
        """Initialize FramesGenerator.

        Args:
            settings: Generator settings.
            register: Modifications register.

        """
        self._setting = settings
        self._register = register or ModificationRegister.default()

    def generate(
        self,
        input_path: pathlib.Path,
    ) -> Generator[ModifiedFrame, None, None]:
        """Generate frames.

        Args:
             input_path: Path to original frames.

        Raises:
            DfdError: when modification for frame could not be retrieved

        Yields:
            modified frames

        """
        if not input_path.is_dir():
            raise DfdError("Input path is not a directory.")
        no_frames = sum(1 for _ in input_path.iterdir())
        for frame_index, input_frame_path in enumerate(input_path.iterdir()):
            modification = self._choose_modification(
                frame_index=frame_index,
                input_frame_path=input_frame_path,
                no_frames=no_frames,
            )
            input_frame = cv.imread(input_frame_path)
            modified_frame = modification.perform(input_frame)
            yield ModifiedFrame(modification_used=str(modification), frame=modified_frame)

    @functools.lru_cache(maxsize=1)
    def _get_modifications_share(self) -> List[ModificationShare]:
        modifications_share: List[ModificationShare] = []
        for modification_settings in self._setting.modifications:
            mame = modification_settings.modification_name
            options = modification_settings.options
            share = modification_settings.share

            modification_class = self._register.get_modification_class(mame)
            # TODO: fix typing
            modification = modification_class(**options)  # type: ignore
            modifications_share.append(ModificationShare(modification, share))
        return modifications_share

    @functools.lru_cache(maxsize=1)
    def _get_modifications_range(self, no_frames: int) -> List[ModificationRange]:
        modifications_share: List[ModificationShare] = self._get_modifications_share()
        modifications_range: List[ModificationRange] = []
        current_summed_share = 0.0
        for modification, share in modifications_share:
            new_summed_share = current_summed_share + share
            modifications_range.append(
                ModificationRange(
                    modification=modification,
                    lower_bound=int(current_summed_share * no_frames),
                    upper_bound=int(new_summed_share * no_frames) - 1,
                )
            )
            current_summed_share = new_summed_share
        # Add identity modification for remaining frames
        modifications_range.append(
            ModificationRange(
                modification=IdentityModification(),
                lower_bound=int(current_summed_share * no_frames),
                upper_bound=no_frames,
            )
        )
        return modifications_range

    @functools.lru_cache(maxsize=1)
    def _get_frames_permutation(self, no_frames: int) -> np.ndarray:
        return np.random.permutation(no_frames)

    def _choose_modification(
        self, frame_index: int, input_frame_path: pathlib.Path, no_frames: int
    ) -> ModificationInterface:
        frames_permutation = self._get_frames_permutation(no_frames)
        modifications_range = self._get_modifications_range(no_frames)
        permuted_index = frames_permutation[frame_index]
        for modification, lower_bound, upper_bound in modifications_range:
            if lower_bound <= permuted_index <= upper_bound:
                return modification
        # TODO: log error
        # This should never happen
        raise DfdError("Could not select modification.")
