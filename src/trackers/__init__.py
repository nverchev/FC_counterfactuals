"""Trackers handling the events of this project.

The trackers for optional dependencies live in their own modules, imported
only when the dependency is installed.
"""

from src.trackers.logging import ClassificationLogger
from src.trackers.setup import get_trackers, set_tuning_logging


__all__ = [
    'ClassificationLogger',
    'get_trackers',
    'set_tuning_logging',
]
