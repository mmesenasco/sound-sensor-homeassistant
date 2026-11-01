# Sound Detector

Listens to a USB microphone plugged into your Home Assistant machine, classifies
audio with Google's YAMNet model, and publishes an event to MQTT whenever one of
the sounds you watch for is heard (glass breaking, alarms, screams, knocks,
speech, dog barks, or your own groups).

Unlike camera-based sound detection (Frigate, yamcam), this reads a **locally
attached microphone**, so it works in rooms with no camera.

## Requirements

- Home Assistant OS or Supervised, on aarch64 or amd64
- A USB microphone plugged into the host
- The **Mosquitto broker** add-on (or any MQTT broker reachable from the host)

A Raspberry Pi 4 runs this easily. Measured on a Pi 4 with Home Assistant OS
18.2: **1.3% CPU and 88 MB RAM**, using the XNNPACK delegate at a 0.5 s hop.

## Installation

1. Add this repository under **Settings → Add-ons → Add-on Store → ⋮ →
   Repositories**.
2. Install **Sound Detector**. The image is built on your machine, which takes
   roughly **9 minutes on a Raspberry Pi 4** and shows no progress in the log.
   That is normal — it is unpacking numpy, scipy and TensorFlow Lite, and the
   time is disk-bound. Watch **Settings → System → Logs → Supervisor** if you
   want to see the build proceed.
3. Plug in the USB microphone **before starting the add-on**.
4. Enable **Watchdog** on the add-on page (see "Recovering from a lost
   microphone" below).
5. Start it, then open the **Log** tab. You want to see
   `YAMNet classifier initialised with N watched class(es)`.

On first start the add-on downloads the YAMNet model (~14 MB) into the image's
`models/` directory if it is not already bundled.

## Picking the microphone

The log prints every input source it can see at startup:

```
Detected audio input sources:
8  alsa_input.usb-Blue_Microphones_Yeti_Nano_...analog-stereo  ...
```

**Leave `mic_device` empty on Home Assistant OS.** Audio reaches this add-on
through Home Assistant's PulseAudio plugin, and PortAudio sees only two devices
through it — `pulse` and `default` — never your microphone by name. Selecting a
specific microphone is done at the Home Assistant level, by making it the
default input:

```
ha audio default input <source-name>     # from the SSH add-on
ha audio info                            # lists source names
```

`mic_device` only applies when running the detector standalone on a normal Linux
box, where PortAudio enumerates ALSA devices directly. There, prefer part of the
device name over an index, since indices shift across reboots and USB
re-enumeration.

If no sources are listed at all, Home Assistant's audio plugin has not picked up
the microphone. Check `ha audio info` from the SSH add-on and confirm the device
appears; a reboot after plugging it in usually resolves it.

## Configuration

| Option | Default | Description |
| --- | --- | --- |
| `device_id` | `barkdetector` | Identifies this sensor in MQTT payloads and discovery |
| `detection_mode` | `yamnet` | `yamnet` (ML) or `heuristic` (RMS + band energy) |
| `watched_events` | `dog_bark,glass_break,alarm,scream` | Built-in sound groups to watch: `dog_bark`, `glass_break`, `alarm`, `scream`, `knock_doorbell` |
| `custom_watched_classes` | *(empty)* | JSON array of your own groups, e.g. `[{"name":"speech","label_substrings":["speech","conversation"]}]` |
| `conf_threshold` | `0.2` | YAMNet confidence required to count a window as a match |
| `normalize_windows` | `false` | Boost quiet windows before inference; helps distant sounds |
| `normalize_noise_floor` | `0.005` | Windows quieter than this are not boosted |
| `normalize_max_gain` | `30` | Ceiling on the boost applied to one window |
| `mic_device` | *(empty)* | Leave empty on HAOS; see "Picking the microphone" above |
| `windows_total` | `5` | Size of the sliding vote window |
| `windows_required` | `2` | Positive windows needed within it to fire an event |
| `cooldown_seconds` | `10` | Minimum gap between events |
| `mqtt_topic` | `home/sensors/dog_bark` | Topic events are published to (default kept from Bark Detector for compatibility) |
| `mqtt_discovery` | `true` | Auto-create the Home Assistant sound sensor |
| `capture_enabled` | `false` | Save a WAV around each event to `/media/barkdetector` |
| `capture_normalize_peak` | `0.7` | Amplify saved clips to this peak so quiet events are audible; `0` saves raw |
| `capture_retention_hours` | `24` | Clips older than this are deleted |
| `capture_max_mb` | `512` | Hard ceiling on the clips folder; oldest go first. `0` disables |
| `log_level` | `info` | Set to `debug` to log the score of every window |

MQTT credentials are taken from the Mosquitto add-on automatically. Only set
`mqtt_host` / `mqtt_port` / `mqtt_username` / `mqtt_password` if you use an
external broker — those manual values take priority when present.

## The entity

With `mqtt_discovery` on, a `sensor` named **Sound** appears under a *Sound
Detector* device. Its **state is the name of the last detected sound**
(`glass_break`, `alarm`, `dog_bark`, or a custom group name) and it keeps that
value until the next event — there is no "off" state. Attributes:

- `score` — YAMNet confidence of the detection (0–1)
- `last_triggered` — epoch seconds of the event; changes on every publish
- `detector` — `yamnet` or `heuristic`

Discovery sets `force_update`, so every MQTT message produces a state change in
Home Assistant even when the same sound repeats.

Example automation (check the exact entity id in your installation):

```yaml
automation:
  - alias: "Notify on glass break"
    triggers:
      - trigger: state
        entity_id: sensor.sound_detector_sound
        to: "glass_break"
    actions:
      - action: notify.notify
        data:
          message: >-
            Glass break detected
            (confidence {{ state_attr('sensor.sound_detector_sound', 'score') }})
```

The raw MQTT payload is also available if you prefer a manual sensor:

```json
{"event": "dog_bark", "score": 0.83, "ts": 1754246400, "device_id": "barkdetector", "detector": "yamnet"}
```

### Disk usage

Each clip is ~320 KB (10 s of 16 kHz mono). Two independent limits apply, and
whichever bites first wins:

- **Age** — `capture_retention_hours` deletes clips older than the window,
  swept hourly.
- **Size** — `capture_max_mb` is a hard ceiling, checked after *every* clip is
  written. When exceeded, the oldest clips are deleted immediately even if
  they are inside the retention window.

The size limit is the one that actually protects the disk: a noisy night can
produce thousands of clips that are all younger than 24 hours, so age alone
bounds nothing. At the 512 MB default that is roughly 1,600 clips.

Clips are written to `/media/barkdetector`, so they appear under **Media → My
media → barkdetector** in the sidebar and play directly in the browser.
Filenames are `YYYYMMDD_HHMMSS_<device_id>.wav` and sort chronologically.

Distant events can be 40 dB below full scale, which is inaudible on playback
even though the detector scores them fine. `capture_normalize_peak` amplifies
each saved clip so its loudest moment reaches the given level. This touches the
saved file only -- detection always runs on the unmodified audio -- so it cannot
affect accuracy. Set it to 0 if you want the raw levels for analysis.

Log files are separately capped at 25 MB (5 MB × 5 rotations).

## Tuning

Start with `log_level: debug` and watch the per-window scores while you play or
make the sound you want to catch.

- **Missed events** — lower `conf_threshold` (try 0.15) or lower
  `windows_required` to 2. Quiet or distant sources score lower.
- **False positives** from speech or TV — raise `conf_threshold` toward 0.4, or
  raise `windows_required`, which demands the sound persist rather than spike.
- **Events arriving in bursts** — raise `cooldown_seconds`. Continuous sources
  such as a person on a call or a TV produce `speech` every few seconds for as
  long as they last; that is correct behaviour, so filter it in your automation
  (e.g. only when nobody is home) or use a long cooldown.

`windows_required` / `windows_total` is the main precision lever, but mind the
arithmetic: with a 0.975 s window and a 0.5 s hop, any instant falls in only
**2** windows. A single short sound (a bark, a knock) can therefore never
satisfy `windows_required: 3` — that setting demands ~1.5 s of sustained sound.
Use 2 to catch one-off events, 3 to reject transients like doors and dropped
cutlery.

### On distant sounds

Detection is limited by the microphone, not by these settings. A sound arriving
below the microphone's self-noise cannot be recovered by gain, thresholds or
normalisation, because all of them scale signal and noise together. If distant
sounds you can clearly hear produce no rise in the `peak` value logged at debug
level, the recording genuinely does not contain them, and the fix is microphone
placement or a more sensitive/directional microphone.

## DailyBot integration (optional)

Set `dailybot_workflow_url` to a [DailyBot](https://dailybot.com/) workflow
trigger URL and each sound event is also POSTed there, for routing to Slack,
Discord or a person. The payload carries the same fields as MQTT plus
`event_type: hardware_sensor` and `secret: sensor`. Leave it empty to disable.

## Recovering from a lost microphone

If the microphone disappears for a moment (USB glitch, power dip, Home
Assistant's audio plugin restarting), PortAudio does not report an error: the
stream simply stops delivering audio. The detector watches for this:

- after **10 s** without audio it closes the stream and reopens the microphone
  (`No audio for ...; reopening the microphone` in the log);
- if audio has still not returned after **120 s**, it exits so the add-on is
  restarted from scratch.

The second step only brings the add-on back if **Watchdog** is enabled on the
add-on page; otherwise it stays stopped and the sensor goes stale.

## Troubleshooting

**The install seems to hang** — it is building, not hung. Allow ~9 minutes on a
Pi 4. The Supervisor log shows the build steps.

**"No MQTT broker available"** — install the Mosquitto broker add-on, or set
`mqtt_host` manually.

**Log says the heuristic detector is active** — YAMNet failed to load; the line
above it says why. The add-on deliberately keeps running on the fallback
detector rather than crashing.

**No audio sources listed** — see "Picking the microphone" above. Note that
Home Assistant's audio plugin holds the ALSA devices, so another add-on reading
`/dev/snd` directly can block this one.

## Credits

Sound Detector is derived from [Bark Detector](https://github.com/maomorales/bark-sensor-home-assistant)
(`barkdetector`) by Mauricio Morales, MIT licensed. The folder name, slug and
MQTT defaults keep the original `barkdetector` naming so that existing
installations update in place.
