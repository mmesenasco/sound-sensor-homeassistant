#!/usr/bin/env python3
"""Debug script to see what YAMNet classifies in real-time audio.

Useful for tuning ``label_substrings`` when adding a new watched class:
run this while making the sound you want to detect, and read off the
exact YAMNet label name(s) it reports.
"""

import numpy as np
import sounddevice as sd
import time
from detector.yamnet import WatchedClass, YAMNetClassifier, YAMNetConfig

SAMPLE_RATE = 16000
DURATION = 0.975  # matches the add-on's default window_seconds

print("🤖 YAMNet Real-time Classification Debug")
print("=" * 70)
print("Shows YAMNet's top-5 overall predictions plus the watched-class match")
print("Press Ctrl+C to stop\n")

config = YAMNetConfig(
    model_url="https://storage.googleapis.com/audioset/yamnet/yamnet.tflite",
    classes_url="https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv",
    watched_classes=[
        WatchedClass("dog_bark", ["dog", "bark", "bow-wow", "yip"], conf_threshold=0.3),
        WatchedClass("glass_break", ["glass", "shatter"], conf_threshold=0.3),
        WatchedClass("alarm", ["alarm", "siren", "smoke detector", "fire alarm"], conf_threshold=0.3),
        WatchedClass("scream", ["scream", "shout", "yell", "crying, sobbing"], conf_threshold=0.3),
    ],
)

try:
    detector = YAMNetClassifier(config)
    print("✅ YAMNet loaded successfully\n")
except Exception as e:
    print(f"❌ Failed to load YAMNet: {e}")
    exit(1)

try:
    while True:
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        audio = audio.flatten()

        predictions = detector._infer(audio)  # noqa: SLF001 - debug script, internal is fine here
        mean_scores = predictions.mean(axis=0)
        top_indices = np.argsort(mean_scores)[::-1][:5]

        matched_label, matched_score = detector.classify(audio)

        if matched_label:
            print(f"\n🔴 MATCH: {matched_label} ({matched_score:.4f})")
        else:
            print("\n   (no watched class above threshold)")

        print("   Top 5 overall predictions:")
        for i, idx in enumerate(top_indices, 1):
            label = detector._all_labels[idx]  # noqa: SLF001
            print(f"   {i}. {label:30s} ({mean_scores[idx]:.4f})")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n\n👋 Stopped")
