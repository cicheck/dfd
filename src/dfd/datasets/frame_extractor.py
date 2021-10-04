"""Extract fom videos frames."""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from .converters import convert_video_to_frames


class FrameExtractor:
    """Extract frames from videos.

    Raw dataset of videos is used to generate directory containing
    frames extracted from original.
    It allows to process dataset gradually, one batch of videos at the time.

    """

    def extract_batch(
        self,
        input_path: Path,
        output_path: Path,
        lower_bound: Optional[int],
        upper_bound: Optional[int],
    ) -> None:
        """Extract frames from batch of videos.

        Split videos into frames and save them into output directory.
        If boundaries are not specified frames_extractor all videos from input directory.
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
        for video in tqdm(processed_input_videos):
            video_frames = convert_video_to_frames(filepath=str(video))
            video_prefix = video.name.split(".")[0]
            for frame_index, frame in enumerate(video_frames):
                frame_path = output_path.joinpath("{0}_{1}.png".format(video_prefix, frame_index))
                self._save_video_frame(frame, str(frame_path))

    @staticmethod
    def _save_video_frame(frame: np.ndarray, filepath: str) -> None:
        cv2.imwrite(filepath, frame)
