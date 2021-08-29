"""Definitions of frame level modifications.

Each modification takes as input single frame and outputs modified frame.

"""

from .clahe import CLAHEModification
from .gamma_correction import GammaCorrectionModification
from .histogram_equalization import HistogramEqualizationModification
from .interfaces import ModificationInterface
