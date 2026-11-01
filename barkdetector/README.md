# Sound Detector

Recognises household sounds — glass breaking, alarms and sirens, screams,
knocks, speech, dog barks — from a **USB microphone plugged into your Home
Assistant machine**, and exposes them as a sensor you can automate on.

Classification uses Google's [YAMNet][yamnet] running locally through TensorFlow
Lite. Nothing leaves your network, and there is no cloud service or account.

## Why this exists

Home Assistant already has good sound detection if you have a camera — [Frigate]
and [yamcam] both classify audio from RTSP streams. Neither helps in a room with
no camera, or when you want a microphone somewhere a camera cannot go.

This add-on reads a locally attached microphone instead, and is meant as a
home-security "ear": know when glass breaks, an alarm goes off or someone is
talking while the house should be empty.

## What you get

- A `sensor` created automatically via MQTT discovery whose state is the last
  detected sound (`glass_break`, `alarm`, `dog_bark`, ...), with `score`,
  `last_triggered` and `detector` attributes
- Built-in sound groups plus your own, defined from YAMNet's 521 labels
- Raw events on an MQTT topic if you prefer to build your own sensor
- Automatic recovery when the microphone stream drops
- Optional audio clips saved to `/media`, playable from the Media panel
- Detection that keeps running on a heuristic fallback if the ML model fails

## Requirements

- Home Assistant OS or Supervised, `aarch64` or `amd64`
- A USB microphone
- The **Mosquitto broker** add-on — credentials are picked up automatically

Light enough to leave running: **1.3% CPU and 88 MB RAM**, measured on a
Raspberry Pi 4.

## Setup

1. Plug in the microphone, then install and start the add-on.
2. Enable **Watchdog** on the add-on page, so it is restarted if it ever exits.
3. Open the **Log** tab and confirm the YAMNet classifier initialised with your
   watched classes.
4. A **Sound** sensor appears under a *Sound Detector* device.

See the **Documentation** tab for configuration, tuning and troubleshooting.

## Microphone choice matters more than any setting

For distant sounds the microphone is the limiting factor, not the software.

Measured on a Raspberry Pi 4 with the same room and settings: a cheap USB
electret recorded **nothing at all** during barks that were clearly audible to a
person, while a USB condenser picked up dogs several hundred metres away and
scored them 0.668. Cheap capsules have a self-noise floor around 35–40 dB SPL, so
a faint sound is not quiet — it is *absent* from the recording.

No amount of gain, threshold or normalisation recovers that, because every one of
them scales signal and noise together. Use the best capsule you can and place it
where the sounds you care about happen.

## Credits

Based on [Bark Detector][upstream] (`barkdetector`) by Mauricio Morales, MIT
licensed, which this add-on extends from dog-bark detection to general sound
events.

[yamnet]: https://www.tensorflow.org/hub/tutorials/yamnet
[Frigate]: https://docs.frigate.video/configuration/audio_detectors/
[yamcam]: https://github.com/cecat/CeC-HA-Addons
[upstream]: https://github.com/maomorales/bark-sensor-home-assistant
