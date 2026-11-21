# MinimapPR Field Mode — Phone Sensors, Timebase and Onboard Mics

As of 2026-11-21. Companion to [field_mode_phone_mesh_plan.md](field_mode_phone_mesh_plan.md), which keeps only a summary of this document.

## Summary

Each phone puts audio, motion sensors and GPS onto one monotonic OS clock, uses each sensor only for what it measures well, and takes fused values from the platform APIs rather than building its own filters. The wall clock (`System.currentTimeMillis`, `Date`) never touches the audio or sensor pipelines, because it jumps whenever the network adjusts it. Onboard multi-mic capture is opportunistic: used only where the OS confirms raw channels and the mic positions come from the OS or a curated per-model table. Users never enter measurements.

## One clock per phone

| Source | Android | iOS native (later) | iOS PWA (interim) |
| --- | --- | --- | --- |
| Shared clock | `CLOCK_BOOTTIME` (`elapsedRealtimeNanos`), keeps ticking in sleep | Mach host time | `performance.now()` |
| Audio | `AudioRecord.getTimestamp(…, TIMEBASE_BOOTTIME)`, or AAudio `getTimestamp(CLOCK_BOOTTIME)` | `AVAudioTime.hostTime` | `AudioContext.getOutputTimestamp()` links context time to `performance.now()` |
| Motion and pressure | `SensorEvent.timestamp` | `CMDeviceMotion.timestamp` | `DeviceOrientationEvent.timeStamp` |
| Location | `Location.getElapsedRealtimeNanos()` | `CLLocation.timestamp` is wall-clock; convert once on arrival | Wall-clock; convert once on arrival |

**Clock anchor.** A single audio timestamp call jitters by 1–5 ms, may return nothing for the first buffers, and carries a fixed, unknown per-model pipeline delay (often 5–40 ms). Each phone therefore fits $t_{\text{boot}} = a + b \cdot n$ (sample index $n$) over a sliding window of timestamp pairs. The fit is smooth to well under a millisecond apart from that fixed delay, which is ample for three jobs:

- placing sensor frames on the audio timeline;
- flagging dropped buffers when residuals jump, as a second check beside sample counting;
- giving the host coarse cross-phone alignment for event matching and clip requests.

The unknown pipeline delay is why it never replaces chirp sync for TDOA.

**GnssClock (Android, Phase 1 measurement).** `GnssMeasurementsEvent` exposes `GnssClock`, which ties GPS time to `CLOCK_BOOTTIME` with no network. Combined with the clock anchor, it gives each phone's audio clock rate against GPS, and UTC offline to within the pipeline-delay uncertainty. Phase 1 measures whether that rate is steady enough to carry a phone's clock model across missed chirp rounds, for example while a phone is briefly in a pocket. It does not replace chirps.

**Sensor frames.** Each phone keeps a small ring buffer of sensor frames at 25–50 Hz beside its PCM buffer, about 1.5 KB/s. A frame holds the time, world orientation quaternion, linear-acceleration level, pressure, and the latest GNSS fix with its quality score. Every clip the host fetches carries the matching slice, interpolated (SLERP for orientation). That way every chirp round and every call has each phone's attitude and motion state at that moment.

## Orientation

- **Android.** `SensorManager` `TYPE_ROTATION_VECTOR` gives a north-referenced orientation with a heading-accuracy estimate. `TYPE_GAME_ROTATION_VECTOR` uses no magnetometer, so it gives smooth short-term rotation that a power bank or trekking pole can't disturb. `GeomagneticField` gives the declination to true north. Both rotation vectors work in a foreground service.
- **Fused Orientation Provider not used.** Google's Play Services provider is cleaner, but [it delivers updates only while the app is in the foreground](https://android-developers.googleblog.com/2024/03/introducing-fused-orientation-provider-api.html), so it can't serve a locked-screen session. The platform rotation vectors are the single source, which avoids jumps from switching between sources.
- **iOS native.** `CMDeviceMotion` with the `.xTrueNorthZVertical` reference frame, which keeps running in the background audio session.
- **Heading matters little.** Magnetometer heading is often 15–40° off near zippers, power banks or ferrous rock. With 3 or more phones, the layout-to-GPS fit sets the group's rotation, so phone heading matters only for stereo phones and couples.

## Motion

`TYPE_STEP_DETECTOR` and `TYPE_LINEAR_ACCELERATION` (gravity already removed) on Android, and `CMPedometer` and `CMDeviceMotion.userAcceleration` on iOS, give two things:

- the walking/still state that schedules chirp rounds;
- a per-frame handling score. Grip shifts and footsteps make broadband transients that line up with acceleration spikes, so the phone blanks those few hundred milliseconds before the Stage-1 trigger, and the host down-weights a clip during them.

The trigger is not switched off while walking, because a solo hiker's notifications on the move are a core use. Positions are never computed by integrating acceleration, which drifts by metres within seconds.

## Pressure

`TYPE_PRESSURE` on Android and `CMAltimeter` on iOS give relative heights to roughly 10–20 cm, but phone barometers can disagree by up to about 1 hPa (8 m) in absolute terms. Each joiner stands next to the host while scanning the join QR code, so the app records the pressure difference at that moment as the phone's offset, good to about the height difference between two people's hands. Weather drift during the session is common to all phones and cancels. GPS altitude is often 15–30 m off and mixes ellipsoid and geoid heights across devices, so it is used only, with a wide uncertainty, for a phone with no barometer.

## GPS

- **Source.** Android `LocationManager.GPS_PROVIDER` feeds the anchor fit, so each fix is an independent GNSS solution. The fused provider's output is already smoothed over time and can mix in Wi-Fi or cell positions. That breaks the independent-errors assumption the multi-phone fit relies on. The fused provider stays in use for the user's own position dot. This choice is provisional: Phase 3 field data compare the two, and the fit switches to the fused provider if it does better.
- **Quality score.** `GnssStatus` reports, per satellite, whether it was used in the fix, its carrier frequency and its C/N0. L5/E5a signals reject forest multipath much better than L1. A fix's weight is its reported accuracy, inflated when few satellites are used, signals are weak or no L5 is locked; under trees the reported accuracy alone is often optimistic. `GnssMeasurement.getMultipathIndicator()` is used where the chipset fills it in.
- **Speed.** GNSS Doppler speed (`Location.getSpeed()`) stays accurate under canopy. It joins the step detector in the "group is still" test that clamps the anchor filter (see Map anchoring in the main plan).
- **iOS.** `CLLocation` only (no raw GNSS), weighted by `horizontalAccuracy`.

## Onboard multi-mic (Android, opportunistic)

Most phones have 2–4 mics (bottom, top, near the rear camera), but the OS often hides them.

- **Never sum raw channels.** Top and bottom mics are about 15 cm apart. An unaligned sum notches out frequencies across the bird band (for a bird off the end of the phone, nulls near 1.1, 3.4, 5.7 and 8 kHz) and turns one arrival into two, which smears every inter-phone TDOA. A HAL that mixes several capsules into its mono output does the same thing and leaves the acoustic centre unknown, so Phase 1 checks which capsule each phone's mono capture uses.
- **Detection at startup.** `AudioRecord.getActiveMicrophones()` returns a `MicrophoneInfo` per mic, with its channel mapping (direct or processed), its position on the phone body in metres, and on some models a frequency-response curve. A phone counts as stereo only if three things hold:
  - two channels are mapped directly;
  - both mic positions are known, from `MicrophoneInfo` or the curated table below;
  - the two channels actually differ, by cross-correlation (many HALs return duplicated mono for a stereo request).

  Otherwise it runs mono.
- **Curated mic table.** When `MicrophoneInfo` leaves positions unknown, which AOSP's default path does unless the vendor supplies geometry, a small per-model table supplies them. It is built from manufacturer dimension drawings, CAD files or teardowns, and we check each entry once with a click played along the phone's long axis (the delay gives the spacing). Over 15 cm, a 5 mm error costs about 2°. It starts with the Pixel models used in testing.
- **Uses.**
  - Bearing: the phone joins the solver as a 2-mic `SubArray` (see Localization solver in the main plan).
  - Chirp headroom: the phone records its own chirp on the mic farthest from the speaker, 12–15 cm away and roughly 15–20 dB quieter, so the chirp level can stay higher.
  - Classifier input: per window, take the channel with less wind energy below 500 Hz. Wind and handling noise are uncorrelated between the mics, while a bird is not.
  - A delay-aligned two-mic sum (up to +3 dB before BirdNET) is a Phase 5 option.

  TDOA always uses one named mic, so the host knows the acoustic centre. Coherence weighting and spectral subtraction are left out, because their artefacts lower BirdNET confidence more than mild noise does.
- **Frequency response.** Where `MicrophoneInfo.getFrequencyResponse()` is populated, it flags phones with a weak high band, which informs the chirp band and band weighting. No EQ is applied before BirdNET unless replay bundles show it helps.
- **iOS.** `.measurement` mode gives unprocessed mono from the primary mic. Stereo polar patterns run Apple's spatial processing, and developers report stereo input dropping to 16 kHz. iPhones stay mono.

## Considered and not adopted

- **Suppressing Stage 1 while walking**, to save battery: it would drop a solo hiker's on-the-move detections. Only handling transients are blanked.
- **Pedestrian dead reckoning** (steps × heading) between chirp rounds: walking phones are already down-weighted, and the anchor filter covers the map. Revisit if rounds prove too sparse while walking.
- **Offline elevation tiles** for absolute ground height: the canopy prior needs only height above local ground, which the phone heights give. Revisit for steep terrain.
- **Cooperative differential GNSS** between Android phones (sharing raw pseudoranges): about 1 m relative positions in open sky, Android-only, and a large custom build. A possible later extension for wide festival deployments where phones are too far apart for chirps.
- **Passive per-phone EQ**, estimating each phone's frequency response by comparing spectra across the mesh: phones hear each source at different distances and angles, so the estimate is fragile.
- **Coherence-weighted or spectral-subtraction denoising** before BirdNET: see Uses above.

## Sources

- [Android Developers Blog: Fused Orientation Provider API (foreground-only updates)](https://android-developers.googleblog.com/2024/03/introducing-fused-orientation-provider-api.html)
- [Android reference: MicrophoneInfo](https://developer.android.com/reference/android/media/MicrophoneInfo)
- [Android CDD 5.4, audio recording (`VOICE_RECOGNITION` with noise reduction and AGC off)](https://android.googlesource.com/platform/compatibility/cdd/+/refs/tags/android-mainline-12.0.0_r98/5_multimedia/5_4_audio-recording.md)
