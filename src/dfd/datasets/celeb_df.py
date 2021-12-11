"""Extract from saved on drive raw Celeb-DF dataset frames (videos -> frames)."""
from pathlib import Path
from typing import Optional

from .frame_extractor import FrameExtractor


# TODO: obsolete faced, should be removed
class CelebDFExtractor:
    """Extract frames from raw celeb-DF.

    Raw dataset of videos is used to generate directory containing
    frames extracted from original videos divided into subdirectories corresponding
    to their classes, i.e. "fake" & "real".
    It allows to process dataset gradually, one batch of videos at the time.

    Input directory must contains directories:
        * Celeb-real: real videos
        * Celeb-synthesis: fake videos

    Created directory has structure:
        * real: directory with frames extracted from real videos
        * fake: directory with frames extracted from fake videos

    """

    def __init__(self, input_path: Path, output_path: Path, frame_extractor: FrameExtractor):
        """Initialize CelebDFExtractor.

        Args:
            input_path: path to raw Celeb-DF dataset.
            output_path: path to preprocessed Celeb-DF dataset.

        """
        self._frame_extractor = frame_extractor

        self._input_path_reals = input_path.joinpath("Celeb-real")
        self._input_path_fakes = input_path.joinpath("Celeb-synthesis")
        self._output_path_reals = output_path.joinpath("reals")
        self._output_path_fakes = output_path.joinpath("fakes")

        # Create subdirectories in output directory
        self._output_path_reals.mkdir(exist_ok=True)
        self._output_path_fakes.mkdir(exist_ok=True)

    def extract_all(self):
        """Preprocess all videos - use at your own risk!

        Preprocess all available real & fake videos.
        If available memory is not sufficient it will crash.

        """
        self.extract_reals_batch()
        self.extract_fakes_batch()

    def extract_reals_batch(
        self,
        lower_bound: Optional[int] = None,
        upper_bound: Optional[int] = None,
    ) -> None:
        """Extract frames from batch of real videos.

        Args:
            lower_bound: lower batch boundary.
            upper_bound: upper batch boundary.

        """
        self._frame_extractor.extract_batch(
            input_path=self._input_path_reals,
            output_path=self._output_path_reals,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def extract_fakes_batch(
        self,
        lower_bound: Optional[int] = None,
        upper_bound: Optional[int] = None,
    ) -> None:
        """Extract frames from batch of fake videos.

        Args:
            lower_bound: lower batch boundary.
            upper_bound: upper batch boundary.

        """
        self._frame_extractor.extract_batch(
            input_path=self._input_path_fakes,
            output_path=self._output_path_fakes,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
