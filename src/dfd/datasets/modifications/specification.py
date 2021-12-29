"""Interfaces used in modifications package."""
from __future__ import annotations

import abc

import numpy as np


class ModificationSpecification(abc.ABC):
    """Base class for modification used to alter frames.

    Design is loosely inspired by specification pattern.

    """

    def __call__(self, image: np.ndarray):
        """Perform modifications, convenience overwrite of dunder method.

        Args:
            image: OpenCV image that will be modified.

        Returns:
            Modified image.

        """
        return self.perform(image)

    def __and__(self, other: ModificationSpecification) -> _AndSpecification:
        return _AndSpecification(self, other)

    @classmethod
    def class_name(cls) -> str:
        """Get class name.

        Returns:
            Specification class name.

        """
        return cls.__name__

    @abc.abstractmethod
    def name(self) -> str:
        """Get specification name.

        Name should include each modification used in specification.

        Returns:
            Specification name.
        """

    @abc.abstractmethod
    def perform(self, image: np.ndarray) -> np.ndarray:
        """Perform modification defined in specification.

        Args:
            image: OpenCV image.

        Returns:
            Modified image.
        """


class _AndSpecification(ModificationSpecification):
    """Combine two specifications."""

    def __init__(
        self, first_spec: ModificationSpecification, sec_spec: ModificationSpecification
    ) -> None:
        self._first_spec = first_spec
        self._sec_spec = sec_spec

    def name(self) -> str:
        """Get specification name.

        Returns:
            Name, combination of names of both specification used to create this one.
        """
        return f"{self._first_spec}__{self._sec_spec}"

    def perform(self, image: np.ndarray) -> np.ndarray:
        """Apply modifications defined in both specifications.

        Modifications are applied in order left -> right.

        Returns:
            Modified image.

        """
        return self._sec_spec(self._first_spec(image))
