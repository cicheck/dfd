from types import MappingProxyType

import pytest

from dfd.exceptions import DfdError
from dfd.models import ModelRegistry
from dfd.models.stubs import ModelStub

REGISTERED_MODELS = MappingProxyType({"stub": ModelStub})


def test_get_registered_model():
    # Given
    registry = ModelRegistry(REGISTERED_MODELS)
    # When
    actual_model_class = registry.get_model_class("stub")
    # Then
    assert actual_model_class is ModelStub


def test_not_registered_model():
    registry = ModelRegistry(REGISTERED_MODELS)
    with pytest.raises(DfdError):
        registry.get_model_class("non_existing_class")
