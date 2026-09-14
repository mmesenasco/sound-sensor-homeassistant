"""Event smoothing and cooldown logic.

Generalised from a single boolean event ("is it a bark") to an optional
label per window ("which watched class matched, if any"). The majority
vote still works the same way -- a label has to recur in enough of the
last ``window_count`` windows before it is trusted -- but the cooldown is
now tracked *per label*, so a glass-break event doesn't have to wait out
a dog-bark cooldown, or vice versa.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional


@dataclass
class SmootherConfig:
    window_count: int
    positives_required: int
    cooldown_seconds: float


@dataclass
class EventSmoother:
    """Implements per-label majority vote with per-label cooldown."""

    config: SmootherConfig
    _history: Deque[Optional[str]] = field(default_factory=deque)
    _scores: Deque[float] = field(default_factory=deque)
    _last_trigger_ts: Dict[str, float] = field(default_factory=dict)
    #: Strongest score across the windows that produced the last trigger. The
    #: score of the final window alone is misleading -- a vote can be carried by
    #: earlier windows, so the triggering window sometimes reads 0.0.
    last_peak_score: float = 0.0

    def update(
        self,
        label: Optional[str],
        timestamp: float | None = None,
        score: float = 0.0,
    ) -> Optional[str]:
        """
        Update the smoother with the latest window's classification.

        Returns the triggered label if this window's vote pushes a label
        over ``positives_required`` within the last ``window_count``
        windows and that label's own cooldown has elapsed, otherwise
        ``None``.
        """
        ts = timestamp if timestamp is not None else time.time()
        cfg = self.config

        if len(self._history) == cfg.window_count:
            self._history.popleft()
            self._scores.popleft()
        self._history.append(label)
        self._scores.append(float(score))

        if label is None:
            return None

        matches = sum(1 for candidate in self._history if candidate == label)
        if matches < cfg.positives_required:
            return None

        last_ts = self._last_trigger_ts.get(label, 0.0)
        if (ts - last_ts) < cfg.cooldown_seconds:
            return None

        self._last_trigger_ts[label] = ts
        relevant_scores = [
            s for candidate, s in zip(self._history, self._scores) if candidate == label
        ]
        self.last_peak_score = max(relevant_scores) if relevant_scores else score
        self._history.clear()
        self._scores.clear()
        return label

    def reset(self) -> None:
        """Clear history and all per-label cooldowns."""
        self._history.clear()
        self._scores.clear()
        self._last_trigger_ts.clear()
        self.last_peak_score = 0.0
