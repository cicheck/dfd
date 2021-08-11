"""Preprocess saved on drive raw Celeb-DF dataset (videos -> frames)."""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from dfd.datasets.converters import convert_video_to_frames


class CelebDFPreprocessor:
    """Preprocess raw celeb-DF.

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

    def __init__(self, input_path: Path, output_path: Path):
        """Initialize CelebDFPreprocessor.

        Args:
            input_path: path to raw Celeb-DF dataset.
            output_path: path to preprocessed Celeb-DF dataset.

        """
        self._input_path_reals = input_path.joinpath("Celeb-real")
        self._input_path_fakes = input_path.joinpath("Celeb-synthesis")
        self._output_path_reals = output_path.joinpath("reals")
        self._output_path_fakes = output_path.joinpath("fakes")

        # Create subdirectories in output directory
        self._output_path_reals.mkdir(exist_ok=True)
        self._output_path_fakes.mkdir(exist_ok=True)

    def preprocess_all(self):
        """Preprocess all videos - use at your own risk!

        Preprocess all available real & fake videos.
        If available memory is not sufficient it will crash.

        """
        self.preprocess_reals_batch()
        self.preprocess_fakes_batch()

    def preprocess_reals_batch(
        self,
        lower_bound: Optional[int] = None,
        upper_bound: Optional[int] = None,
    ) -> None:
        """Preprocess batch of real videos.

        Args:
            lower_bound: lower batch boundary.
            upper_bound: upper batch boundary.

        """
        self._preprocess_videos_batch(
            input_path=self._input_path_reals,
            output_path=self._output_path_reals,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def preprocess_fakes_batch(
        self,
        lower_bound: Optional[int] = None,
        upper_bound: Optional[int] = None,
    ) -> None:
        """Preprocess batch of fake videos.

        Args:
            lower_bound: lower batch boundary.
            upper_bound: upper batch boundary.

        """
        self._preprocess_videos_batch(
            input_path=self._input_path_fakes,
            output_path=self._output_path_fakes,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def _preprocess_videos_batch(
        self,
        input_path: Path,
        output_path: Path,
        lower_bound: Optional[int],
        upper_bound: Optional[int],
    ) -> None:
        """Preprocess batch of videos.

        Split videos into frames and save them into output directory.
        If boundaries are not specified preprocess all videos from input directory.
        Frames are saved in files named by number in which they were produced.
        Starting number is determined by number of files already existing in directory.

        Args:
            input_path: path to directory containing videos.
            output_path: path to directory where frames from videos should be saved.
            lower_bound: lower batch boundary.
            upper_bound: upper batch boundary.

        """
        all_input_videos = sorted(input_path.iterdir())
        processed_input_videos = all_input_videos[lower_bound:upper_bound]
        for video in processed_input_videos:
            video_frames = convert_video_to_frames(filepath=str(video))
            video_prefix = video.name.split(".")[0]
            for frame_index, frame in enumerate(video_frames):
                frame_path = output_path.joinpath("{0}_{1}.jpg".format(video_prefix, frame_index))
                self._save_video_frame(frame, str(frame_path))

    @staticmethod
    def _save_video_frame(frame: np.ndarray, filepath: str) -> None:
        cv2.imwrite(filepath, frame)
