"""YAMNet-based multi-class sound detector.

Originally this module only ever answered "is this a bark, yes or no?"
by collapsing every matching YAMNet class into a single confidence score.
This version keeps the same model and the same config-driven substring
matching, but tracks *several* named "watched classes" independently and
reports which one (if any) fired on a given audio window, so a single
sensor can report "Dog bark", "Glass, Shatter", "Siren", etc.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import requests
from loguru import logger

# Imported lazily-ish: a missing interpreter must not break `import detector.yamnet`,
# or main.py dies at import time and the heuristic fallback never gets a chance.
# tflite-runtime is frozen at 2.14.0 (no wheel past CPython 3.11), so newer
# interpreters need its successor package, ai-edge-litert.
try:
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except ImportError:  # pragma: no cover - depends on the installed runtime
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
    except ImportError:
        Interpreter = None  # type: ignore[assignment]


class YAMNetInitializationError(Exception):
    """Raised when the YAMNet detector cannot be initialised."""


@dataclass
class WatchedClass:
    """One named group of YAMNet labels to watch for, with its own threshold.

    ``name`` is used both as the published event slug (e.g. "glass_break")
    and, unless overridden, is what ends up in the MQTT payload/HA sensor
    state. Keep it short and stable -- automations will match on it.
    """

    name: str
    label_substrings: Iterable[str]
    conf_threshold: float


@dataclass
class YAMNetConfig:
    model_url: str
    classes_url: str
    watched_classes: List[WatchedClass] = field(default_factory=list)


class YAMNetClassifier:
    """Wraps a TFLite YAMNet model and scores a fixed set of watched classes."""

    def __init__(self, config: YAMNetConfig, models_dir: Path | None = None) -> None:
        self.config = config
        self.models_dir = models_dir or Path(__file__).resolve().parents[1] / "models"
        self.model_path = self.models_dir / "yamnet.tflite"
        self.classes_path = self.models_dir / "yamnet_class_map.csv"
        if Interpreter is None:
            raise YAMNetInitializationError(
                "No TFLite interpreter available. Install 'tflite-runtime' "
                "(CPython <= 3.11) or 'ai-edge-litert' (CPython >= 3.12)."
            )

        self._interpreter: Interpreter | None = None
        # name -> list of YAMNet class indices matching that watched class
        self._class_indices: dict[str, List[int]] = {}
        self._thresholds: dict[str, float] = {}
        self._input_index: int | None = None
        self._output_index: int | None = None
        self._current_input_length: int | None = None
        self._all_labels: List[str] = []

        try:
            self._prepare_files()
            self._load_interpreter()
            self._load_class_map()
        except Exception as exc:  # pragma: no cover - defensive
            raise YAMNetInitializationError(str(exc)) from exc

    def classify(self, samples: np.ndarray) -> Tuple[Optional[str], float]:
        """Score every watched class on this window.

        Returns ``(name, score)`` for the highest-scoring watched class that
        clears *its own* threshold, or ``(None, 0.0)`` if none did. When
        several watched classes clear their threshold in the same window,
        the one with the highest score wins -- a window is reported as one
        event, not several, to keep downstream smoothing/MQTT simple.
        """
        predictions = self._infer(samples)

        best_name: Optional[str] = None
        best_score = -1.0  # allow a legitimate score of exactly 0.0 to win when threshold is 0.0 too
        for name, indices in self._class_indices.items():
            if not indices:
                continue
            score = float(np.max(predictions[:, indices])) if predictions.size else 0.0
            score = float(np.clip(score, 0.0, 1.0))
            threshold = self._thresholds.get(name, 1.0)
            if score >= threshold and score > best_score:
                best_name = name
                best_score = score

        return best_name, best_score

    def top_label(self, samples: np.ndarray) -> Tuple[str, float]:
        """Return YAMNet's single highest-scoring label overall, for debugging.

        Not used by the detection pipeline -- handy when tuning
        ``label_substrings`` for a new watched class, to see what YAMNet
        actually calls a sound you just made.
        """
        predictions = self._infer(samples)
        mean_scores = predictions.mean(axis=0) if predictions.size else np.zeros(len(self._all_labels))
        if mean_scores.size == 0:
            return "unknown", 0.0
        idx = int(np.argmax(mean_scores))
        label = self._all_labels[idx] if idx < len(self._all_labels) else "unknown"
        return label, float(mean_scores[idx])

    # Internal helpers -------------------------------------------------

    def _infer(self, samples: np.ndarray) -> np.ndarray:
        if self._interpreter is None or self._input_index is None or self._output_index is None:
            raise RuntimeError("YAMNet interpreter is not initialised")

        waveform = samples.astype(np.float32, copy=False)
        if waveform.ndim != 1:
            waveform = waveform.squeeze()

        if self._current_input_length != waveform.shape[0]:
            self._resize_input(len(waveform))

        self._interpreter.set_tensor(self._input_index, waveform)
        self._interpreter.invoke()
        return self._interpreter.get_tensor(self._output_index)

    def _prepare_files(self) -> None:
        os.makedirs(self.models_dir, exist_ok=True)
        if not self.model_path.exists():
            logger.info("Downloading YAMNet model from {}", self.config.model_url)
            self._download_file(self.config.model_url, self.model_path)
        if not self.classes_path.exists():
            logger.info("Downloading YAMNet class map from {}", self.config.classes_url)
            self._download_file(self.config.classes_url, self.classes_path)

    def _download_file(self, url: str, destination: Path) -> None:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        tmp_path = destination.with_suffix(destination.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
        tmp_path.replace(destination)

    def _load_interpreter(self) -> None:
        self._interpreter = Interpreter(model_path=str(self.model_path))
        self._interpreter.allocate_tensors()
        input_details = self._interpreter.get_input_details()[0]
        output_details = self._interpreter.get_output_details()[0]
        self._input_index = int(input_details["index"])
        self._output_index = int(output_details["index"])
        self._current_input_length = input_details["shape"][0] or None

    def _resize_input(self, length: int) -> None:
        if self._interpreter is None or self._input_index is None:
            raise RuntimeError("Interpreter not prepared")
        self._interpreter.resize_tensor_input(self._input_index, [length], strict=False)
        self._interpreter.allocate_tensors()
        self._current_input_length = length

    def _load_class_map(self) -> None:
        with self.classes_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            labels = [row.get("display_name", "").strip() for row in reader]

        if not labels:
            raise RuntimeError("YAMNet class map is empty")
        self._all_labels = labels

        if not self.config.watched_classes:
            logger.warning("No watched_classes configured; detector will never trigger")

        for watched in self.config.watched_classes:
            substrings = [s.lower() for s in watched.label_substrings]
            indices = [
                idx for idx, label in enumerate(labels)
                if any(sub in label.lower() for sub in substrings)
            ]
            if not indices:
                logger.warning(
                    "Watched class '{}' matched no YAMNet labels for substrings {}; "
                    "it will never trigger",
                    watched.name,
                    substrings,
                )
            else:
                logger.info(
                    "Watched class '{}' -> {} YAMNet label(s): {}",
                    watched.name,
                    len(indices),
                    [labels[i] for i in indices],
                )
            self._class_indices[watched.name] = indices
            self._thresholds[watched.name] = watched.conf_threshold


# Backwards-compatible alias: existing imports/tests referencing the old
# single-class name keep working.
YAMNetBarkDetector = YAMNetClassifier
