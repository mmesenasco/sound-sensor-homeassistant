"""Detector package for multi-class sound detection components."""

from .yamnet import (
    WatchedClass,
    YAMNetBarkDetector,  # backwards-compatible alias for YAMNetClassifier
    YAMNetClassifier,
    YAMNetConfig,
    YAMNetInitializationError,
)
from .heuristic import HeuristicBarkDetector
from .smoothing import EventSmoother
from .audio import AudioStreamConfig, AudioStreamProvider
