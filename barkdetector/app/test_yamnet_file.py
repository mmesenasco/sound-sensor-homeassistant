#!/usr/bin/env python3
"""Test YAMNet on a captured WAV file, reporting which watched class matched."""

import sys
import wave
import numpy as np
from detector.yamnet import WatchedClass, YAMNetClassifier, YAMNetConfig

if len(sys.argv) < 2:
    print("Usage: python3 test_yamnet_file.py <wav_file>")
    sys.exit(1)

wav_file = sys.argv[1]

config = YAMNetConfig(
    model_url="https://storage.googleapis.com/audioset/yamnet/yamnet.tflite",
    classes_url="https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv",
    watched_classes=[
        WatchedClass("dog_bark", ["dog", "bark", "bow-wow", "yip"], conf_threshold=0.01),
        WatchedClass("glass_break", ["glass", "shatter"], conf_threshold=0.01),
        WatchedClass("alarm", ["alarm", "siren", "smoke detector", "fire alarm"], conf_threshold=0.01),
        WatchedClass("scream", ["scream", "shout", "yell", "crying, sobbing"], conf_threshold=0.01),
    ],
)

print("Loading YAMNet...")
detector = YAMNetClassifier(config)
print("✅ YAMNet loaded")
for name, indices in detector._class_indices.items():  # noqa: SLF001 - debug script
    print(f"   Watched class '{name}': {len(indices)} matching YAMNet label(s)")

print(f"\nLoading WAV file: {wav_file}")
with wave.open(wav_file, "rb") as wf:
    sample_rate = wf.getframerate()
    n_frames = wf.getnframes()
    audio_bytes = wf.readframes(n_frames)
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

print(f"   Sample rate: {sample_rate} Hz")
print(f"   Duration: {len(audio) / sample_rate:.2f} seconds")
print(f"   Audio range: [{audio.min():.3f}, {audio.max():.3f}]")

window_size = 15600  # ~0.975s at 16kHz, matches the add-on default
hop_size = 8000

print("\nProcessing audio in chunks...")
best_label = None
best_score = 0.0
for i in range(0, max(len(audio) - window_size, 0), hop_size):
    chunk = audio[i : i + window_size]
    label, score = detector.classify(chunk)
    if label and score > 0.0:
        print(f"   Time {i/sample_rate:.2f}s: {label} = {score:.4f}")
    if score > best_score:
        best_score = score
        best_label = label

print(f"\n📊 Best match: {best_label or '(none)'} (score={best_score:.4f})")
if best_label:
    print(f"✅ DETECTED: {best_label}")
else:
    print("❌ No watched class matched (all scores stayed under threshold)")
