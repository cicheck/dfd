"""Modification Gaussian noise."""

import numpy as np

from dfd.datasets.modifications.specification import ModificationSpecification


class GaussianNoiseModification(ModificationSpecification):
    """Modification Gaussian noise."""

    def __init__(
        self,
        mean: float = 0.0,
        standard_deviation: float = 1.0,
    ) -> None:
        """Initialize GaussianNoiseModification.

        Args:
            mean: Mean of Gaussian distribution used to draw sample from.
            standard_deviation: Standard deviation of Gaussian distribution
                used to draw sample from.

        """
        self._mean = mean
        self._standard_deviation = standard_deviation

    @property
    def name(self) -> str:
        """Get specification name.

        Returns:
            The name of specification.

        """
        return f"gaussian_noise{self._mean}_{self._standard_deviation}"

    def perform(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to provided image.

        Args:
            image: OpenCV image.

        Returns:
            Image with Gaussian noise added.

        """
        drawn_samples = np.random.normal(self._mean, self._standard_deviation, image.shape)
        noise = drawn_samples.reshape(image.shape)
        modified_image = image + noise
        # Convert returned image to uint8 since float values are valid for noise added
        return modified_image.astype("uint8")
