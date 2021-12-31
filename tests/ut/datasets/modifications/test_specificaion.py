import functools
import operator
import typing as t
from unittest.mock import Mock, call

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from dfd.datasets.modifications.specification import ModificationSpecification


class ModificationStub(ModificationSpecification):
    def __init__(self, name: str, no_repeats: int) -> None:
        self._name = name
        self.no_repeats = no_repeats

    @property
    def name(self) -> str:
        return self._name

    def perform(self, image: np.ndarray) -> np.ndarray:
        image.repeat(self.no_repeats)
        return image


@given(
    given_specifications=st.lists(
        st.builds(
            ModificationStub,
            name=st.text(min_size=1),
            no_repeats=st.integers(min_value=1),
        ),
        min_size=1,
        max_size=5,
    )
)
def test_combine_multiple_specifications(given_specifications):
    # GIVEN
    image_mock = Mock(spec_set=np.ndarray)
    # WHEN specifications are combined
    combined_specification = functools.reduce(operator.and_, given_specifications)
    # THEN specification names are combined
    expected_name = "__".join([spec.name for spec in given_specifications])
    assert combined_specification.name == expected_name
    # And specifications are performed in order
    combined_specification.perform(image_mock)
    expected_calls_in_order = [call.repeat(spec.no_repeats) for spec in given_specifications]
    image_mock.assert_has_calls(expected_calls_in_order)
