"""Audio capture utilities for sound-event detection."""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

import numpy as np
import sounddevice as sd
from loguru import logger
from scipy.signal import resample_poly


@dataclass
class AudioStreamConfig:
    """Configuration for the audio stream."""

    sample_rate: int
    channels: int
    window_seconds: float
    hop_seconds: float
    # An int selects a PortAudio device index, a str matches part of a device
    # name (indices are not stable across reboots or USB re-enumeration), and
    # None uses the system default input.
    mic_device_index: Optional[object] = None


class AudioStreamError(Exception):
    """Raised when the audio stream cannot be established."""


# If no audio arrives for this long, the stream is considered dead and is
# reopened. PortAudio does not raise when the ALSA/PulseAudio side goes away:
# the callback simply stops firing, so without this the loop waits forever.
STALL_SECONDS = 10.0

# Last resort: if nothing has been yielded for this long (e.g. closing the dead
# stream itself hangs inside PortAudio), kill the process so s6/Supervisor
# restart the add-on instead of leaving it silently stuck.
HARD_STALL_SECONDS = 120.0


class AudioStreamProvider:
    """Provides resampled audio chunks ready for detection."""

    def __init__(self, config: AudioStreamConfig) -> None:
        self.config = config
        self._stop_event = threading.Event()
        self._source_rate: Optional[int] = None
        self._last_audio = time.monotonic()
        self._watchdog: Optional[threading.Thread] = None

    @staticmethod
    def list_input_devices() -> List[Tuple[int, str, float, int]]:
        """Return a list of available input devices."""
        devices = sd.query_devices()
        result: List[Tuple[int, str, float, int]] = []
        for idx, info in enumerate(devices):
            if info.get("max_input_channels", 0) > 0:
                result.append(
                    (
                        idx,
                        info.get("name", f"Device {idx}"),
                        float(info.get("default_samplerate", 0) or 0),
                        int(info.get("max_input_channels", 0)),
                    )
                )
        return result

    def stop(self) -> None:
        """Signal the audio stream to stop."""
        self._stop_event.set()

    def _start_watchdog(self) -> None:
        """Exit the process if audio stays stalled past HARD_STALL_SECONDS."""
        if self._watchdog is not None:
            return

        def run() -> None:
            while not self._stop_event.wait(5.0):
                idle = time.monotonic() - self._last_audio
                if idle > HARD_STALL_SECONDS:
                    logger.critical(
                        "No audio for {:.0f}s and the stream did not recover; exiting so the add-on restarts",
                        idle,
                    )
                    os._exit(1)

        self._watchdog = threading.Thread(target=run, name="audio-watchdog", daemon=True)
        self._watchdog.start()

    def stream_chunks(self) -> Generator[np.ndarray, None, None]:
        """
        Yield audio chunks resampled to target sample rate.

        Each yielded chunk spans hop_seconds worth of audio at the target
        sample rate (e.g., 0.5 seconds -> 8000 samples at 16 kHz).
        """
        target_rate = self.config.sample_rate
        hop_samples_target = max(1, int(round(self.config.hop_seconds * target_rate)))

        self._last_audio = time.monotonic()
        self._start_watchdog()

        while not self._stop_event.is_set():
            queue_: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)

            try:
                stream, source_rate = self._open_stream(queue_)
            except AudioStreamError as exc:
                logger.error("Audio stream setup failed: {}", exc)
                time.sleep(5.0)
                continue

            self._source_rate = source_rate

            logger.info(
                "Audio stream started at {} Hz (target {} Hz) using device {}",
                source_rate,
                target_rate,
                self.config.mic_device_index
                if self.config.mic_device_index is not None
                else "default",
            )

            try:
                with stream:
                    stream_last_audio = time.monotonic()
                    while not self._stop_event.is_set():
                        try:
                            chunk = queue_.get(timeout=1.0)
                        except queue.Empty:
                            stalled = time.monotonic() - stream_last_audio
                            if not stream.active or stalled > STALL_SECONDS:
                                logger.warning(
                                    "No audio for {:.1f}s (stream active={}); reopening the microphone",
                                    stalled,
                                    stream.active,
                                )
                                break
                            continue

                        stream_last_audio = time.monotonic()
                        self._last_audio = stream_last_audio

                        if chunk.size == 0:
                            continue

                        mono = self._to_mono(chunk)

                        if source_rate != target_rate:
                            resampled = resample_poly(mono, target_rate, source_rate)
                        else:
                            resampled = mono

                        if resampled.size == 0:
                            continue

                        if resampled.size != hop_samples_target:
                            # Pad or trim to maintain consistent chunk size.
                            resampled = self._pad_or_trim(resampled, hop_samples_target)

                        yield resampled.astype(np.float32, copy=False)
            except sd.PortAudioError as exc:
                logger.error("Audio stream encountered an error: {}", exc)
                time.sleep(2.0)
            finally:
                logger.info("Audio stream stopped; reconnecting in 2 seconds")
                time.sleep(2.0)

    def _resolve_device(self) -> Optional[int]:
        """Turn the configured device selector into a PortAudio index."""
        selector = self.config.mic_device_index
        if selector is None or isinstance(selector, int):
            return selector

        needle = str(selector).strip().lower()
        if not needle:
            return None

        for idx, name, _rate, _channels in self.list_input_devices():
            if needle in name.lower():
                logger.info("Matched microphone '{}' to device [{}] {}", selector, idx, name)
                return idx

        available = ", ".join(
            f"[{idx}] {name}" for idx, name, _r, _c in self.list_input_devices()
        ) or "none"
        logger.error(
            "No input device matches '{}'; falling back to the default. Available: {}",
            selector,
            available,
        )
        return None

    def _open_stream(
        self, queue_: "queue.Queue[np.ndarray]"
    ) -> Tuple[sd.InputStream, int]:
        device = self._resolve_device()
        target_rate = self.config.sample_rate
        candidate_rates: List[int] = [target_rate]

        try:
            device_info = sd.query_devices(device, "input")
            default_rate = int(device_info.get("default_samplerate", 0) or 0)
            if default_rate and default_rate not in candidate_rates:
                candidate_rates.append(default_rate)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Unable to query device info for {}: {}", device, exc)

        for rate in candidate_rates:
            try:
                hop_samples = max(1, int(round(self.config.hop_seconds * rate)))
                stream = sd.InputStream(
                    device=device,
                    samplerate=rate,
                    blocksize=hop_samples,
                    channels=self.config.channels,
                    dtype="float32",
                    callback=self._make_callback(queue_),
                )
                return stream, rate
            except sd.PortAudioError as exc:
                logger.warning(
                    "Failed to open audio stream at {} Hz (device {}): {}",
                    rate,
                    device,
                    exc,
                )

        raise AudioStreamError("Unable to open microphone stream at any supported rate")

    @staticmethod
    def _make_callback(
        queue_: "queue.Queue[np.ndarray]",
    ) -> "sd.CallbackType":
        def callback(indata, frames, time_info, status):  # type: ignore[override]
            if status:
                logger.warning("Audio callback status: {}", status)
            try:
                queue_.put_nowait(indata.copy())
            except queue.Full:
                # Drop oldest sample to avoid blocking callback.
                try:
                    queue_.get_nowait()
                    queue_.put_nowait(indata.copy())
                except queue.Empty:
                    pass

        return callback

    def _to_mono(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.ndim == 1 or chunk.shape[1] == 1:
            return np.squeeze(chunk).astype(np.float32, copy=False)
        return np.mean(chunk, axis=1).astype(np.float32, copy=False)

    @staticmethod
    def _pad_or_trim(samples: np.ndarray, target_size: int) -> np.ndarray:
        if samples.size == target_size:
            return samples
        if samples.size > target_size:
            return samples[:target_size]
        pad_width = target_size - samples.size
        return np.pad(samples, (0, pad_width), mode="constant")
