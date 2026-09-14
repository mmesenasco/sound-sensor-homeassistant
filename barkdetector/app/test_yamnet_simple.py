#!/usr/bin/env python3
"""Simple real-time monitor for the watched-class match score."""

import sounddevice as sd
import time
from detector.yamnet import WatchedClass, YAMNetClassifier, YAMNetConfig

SAMPLE_RATE = 16000
DURATION = 0.975  # matches the add-on's default window_seconds
THRESHOLD = 0.3

print("🤖 YAMNet Watched-Class Monitor")
print("=" * 60)
print(f"Threshold: {THRESHOLD}")
print("Press Ctrl+C to stop\n")

config = YAMNetConfig(
    model_url="https://storage.googleapis.com/audioset/yamnet/yamnet.tflite",
    classes_url="https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv",
    watched_classes=[
        WatchedClass("dog_bark", ["dog", "bark", "bow-wow", "yip"], conf_threshold=THRESHOLD),
        WatchedClass("glass_break", ["glass", "shatter"], conf_threshold=THRESHOLD),
        WatchedClass("alarm", ["alarm", "siren", "smoke detector", "fire alarm"], conf_threshold=THRESHOLD),
        WatchedClass("scream", ["scream", "shout", "yell", "crying, sobbing"], conf_threshold=THRESHOLD),
    ],
)

try:
    detector = YAMNetClassifier(config)
    print("✅ YAMNet loaded successfully")
    for name, indices in detector._class_indices.items():  # noqa: SLF001 - debug script
        print(f"   '{name}': {len(indices)} matching label(s)")
    print()
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

        label, score = detector.classify(audio)

        status = f"🔴 {label.upper()}!" if label else "⚪ Listening..."
        bar_len = int(score * 50)
        bar = "█" * bar_len + "░" * (50 - bar_len)

        print(f"{status:20s} | Score: {score:.4f} | [{bar}]")

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\n\n👋 Stopped")
