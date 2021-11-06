"""Concrete implementation of ML models.

Code outside `models` package shouldn't depend on concrete implementations available
in this package. Instead higher level interface should be used to reduce coupling.

"""

from .meso_net import MesoNet
