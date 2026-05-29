"""
Torchpac
=========

Torchpac is an open-source Python toolbox designed for computing
Phase-Amplitude Coupling.
"""
import logging

from torchpac import methods, signals, utils, stats  # noqa
from torchpac.pac import (Pac, EventRelatedPac, PreferredPhase)  # noqa
from torchpac.io import set_log_level
# Set 'info' as the default logging level
logger = logging.getLogger('brainets')
set_log_level('info')

__version__ = "0.6.5"
