"""Extract from raw datasets stored on disk frames.

Resulting frames datasets contains directories:
    * reals: frames from real videos
    * fakes: frames from fake videos

Each frame filename follow schema {video_name}.{frame_index}.{extension}

"""

from .celeb_df import CelebDFExtractor
