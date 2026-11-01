# Sound Detector — Home Assistant add-on

Recognise household sounds — **glass breaking, alarms and sirens, screams,
knocks, speech, dog barks** — from a **USB microphone attached to your Home
Assistant machine**, and turn them into a sensor you can automate on.

Audio is classified locally by Google's [YAMNet][yamnet] via TensorFlow Lite.
Nothing leaves your network, and there is no cloud service or account.

[![Open your Home Assistant instance and show the add add-on repository dialog.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https://github.com/mmesenasco/sound-sensor-homeassistant)

## Install

**Settings → Add-ons → Add-on Store → ⋮ → Repositories**, and add:

```
https://github.com/mmesenasco/sound-sensor-homeassistant
```

Then install **Sound Detector**. Full documentation is in
[`barkdetector/DOCS.md`](barkdetector/DOCS.md).

> **The first install builds the image on your machine and takes a while** —
> about 9 minutes on a Raspberry Pi 4 with a class-10 SD card. The add-on log
> stays quiet during this; it is not stuck. Most of the time goes on unpacking
> the numpy, scipy and TensorFlow Lite wheels, which is disk-bound rather than
> CPU-bound. Later updates reuse cached layers and are much faster.

## Why a local microphone

[Frigate][frigate] and [yamcam][yamcam] classify audio from **camera RTSP
streams**, and both work well. This add-on reads a **locally attached
microphone**, for rooms with no camera or spots a camera cannot reach — the
typical use being a home-security "ear" that tells you when something happens
while nobody is home.

## Features

- Multi-class YAMNet (TFLite) classification: built-in groups (`dog_bark`,
  `glass_break`, `alarm`, `scream`, `knock_doorbell`) plus any custom group you
  define from YAMNet's 521 labels
- A single MQTT-discovered `sensor` whose state is the last detected sound, with
  `score` and `last_triggered` attributes — one entity, one automation per sound
- MQTT credentials taken from the Mosquitto add-on via the Supervisor
- Sliding-window voting with a cooldown, to reject one-off transients
- Recovers by itself when the microphone stream dies (USB glitch, audio plugin
  restart) instead of going silently deaf
- Heuristic fallback if the model fails to load, so detection degrades rather
  than stops
- Optional clips saved to `/media`, playable in the Media panel, with the
  confidence score in the filename; storage bounded by both age and total size

Resource use on a Raspberry Pi 4: **1.3% CPU, 88 MB RAM**.

## Example automation

```yaml
triggers:
  - trigger: state
    entity_id: sensor.sound_detector_sound   # check the exact id in your HA
    to: "glass_break"
actions:
  - action: notify.notify
    data:
      message: >-
        🚨 Glass break detected (confidence
        {{ state_attr('sensor.sound_detector_sound', 'score') }})
```

More examples, including the raw MQTT form, in
[`ha_automation_example.yaml`](ha_automation_example.yaml).

## The microphone is the hard part

If the sounds you care about are far away, **the microphone determines whether
this works** — not the software settings.

Measured on the same Pi, in the same room, with identical settings:

| | Cheap USB electret | USB condenser |
| --- | --- | --- |
| Idle noise floor (peak) | 0.0115 | 0.0055 |
| Distant barks clearly audible to a person | **nothing recorded** | detected, score 0.668 |

Cheap capsules have a self-noise floor near 35–40 dB SPL. A distant bark is not
merely quiet, it is *below that floor* — absent from the recording entirely. Gain,
thresholds and normalisation all scale signal and noise together, so none of them
recover it.

What actually helps, in order of impact:

1. **Move the microphone outdoors or to an open window** — worth 20–30 dB, more
   than every software option combined
2. **Use a low-self-noise capsule** — a condenser rather than a cheap electret
3. **Point a directional mic at the source** — roughly 6 dB, and unlike gain it
   genuinely improves the ratio by rejecting noise the signal does not share

Set expectations accordingly: dogs in your house or a neighbour's yard are
reliable. A kilometre away, outdoors, on a quiet night is best-effort.

## Running standalone

The detector also runs on any Linux box without Home Assistant. The application
lives in [`barkdetector/app/`](barkdetector/app).

```bash
git clone https://github.com/mmesenasco/sound-sensor-homeassistant.git
cd sound-sensor-homeassistant/barkdetector/app
./scripts/setup.sh
cp config/example-config.yaml config/config.yaml   # then edit
python3 main.py --config config/config.yaml
```

Requires Python 3.10–3.11: `tflite-runtime` publishes no wheel past CPython 3.11.
On 3.12+ install `ai-edge-litert` instead, which the code also accepts.

MQTT credentials can be supplied as `BARKDETECTOR_MQTT_HOST` / `_PORT` /
`_TOPIC` / `_USERNAME` / `_PASSWORD`, which override the config file — keep them
out of version control.

Useful flags:

```bash
python3 main.py --list-devices             # find your microphone
python3 main.py --config ... --dry-run     # detect without publishing
```

## Repository layout

```
repository.yaml          Home Assistant add-on repository manifest
barkdetector/            the add-on (folder and slug keep the original name,
                         so existing installs update in place)
  config.yaml            options schema and add-on metadata
  Dockerfile             Debian-based image (see note below)
  build.yaml             per-architecture base images
  rootfs/                s6 service and options-to-config renderer
  app/                   the Python application
```

### Why Debian and not the default Alpine base

`numpy`, `scipy` and `tflite-runtime` publish manylinux (glibc) wheels only. On
Alpine's musl, pip falls back to building from source and fails on a Pi. The base
is pinned to **bookworm** specifically, because bookworm ships Python 3.11 and
`tflite-runtime` 2.14.0 — its final release — has no wheel beyond cp311.

`numpy` is pinned below 2.0 for the same reason: `tflite-runtime` is compiled
against the NumPy 1.x C ABI, and NumPy 2.x makes the interpreter fail to load
with `_ARRAY_API not found`.

## Credits

Sound Detector started from **[Bark Detector][upstream]** (`barkdetector`) by
Mauricio Morales, a Home Assistant add-on that detects dog barks with YAMNet. Its
add-on structure, audio pipeline, MQTT integration, clip capture and much of the
microphone research in this README come from that project — thank you.

This fork extends it into a general sound-event sensor for home security:
multi-class and custom sound groups, a single multi-state sensor in place of the
bark `binary_sensor`, repeat-event handling, and automatic recovery from a lost
microphone stream. The internal folder, slug and MQTT defaults still carry the
`barkdetector` name so that existing installations keep working.

## Licence

MIT — see [LICENSE](LICENSE). Copyright is shared between the original author of
Bark Detector and the maintainer of this fork.

[yamnet]: https://www.tensorflow.org/hub/tutorials/yamnet
[frigate]: https://docs.frigate.video/configuration/audio_detectors/
[yamcam]: https://github.com/cecat/CeC-HA-Addons
[upstream]: https://github.com/maomorales/bark-sensor-home-assistant
