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

**Overrun counter.** With AAudio capture, `AAudioStream_getXRunCount()` counts input overruns directly. Any rise marks a gap at once, as a third check beside the sample count and the clock-anchor residual. `AudioRecord` has no input equivalent, which is one reason Phase 1 compares the two capture paths. AAudio's power-saving mode (larger buffers, fewer wakeups) also suits a design whose timing comes from sample counts rather than callback arrival, provided Phase 1 finds no hidden resampling or stutter on the target Pixels.

**GnssClock (Android, Phase 1 measurement).** `GnssMeasurementsEvent` exposes `GnssClock`, which ties GPS time to `CLOCK_BOOTTIME` with no network. Combined with the clock anchor, it gives each phone's audio clock rate against GPS, and UTC offline to within the pipeline-delay uncertainty. Phase 1 measures whether that rate is steady enough to carry a phone's clock model across missed chirp rounds, for example while a phone is briefly in a pocket. It does not replace chirps.

**Sensor frames.** Each phone keeps a small ring buffer of sensor frames at 25–50 Hz beside its PCM buffer, about 1.5 KB/s. A frame holds the time, world orientation quaternion, linear-acceleration level, pressure, and the latest GNSS fix with its quality score. Every clip the host fetches carries the matching slice, interpolated (SLERP for orientation). That way every chirp round and every call has each phone's attitude and motion state at that moment.

**Hub batching.** Registering these sensors with the default `maxReportLatencyUs = 0` wakes the application processor 25–50 times a second. Instead they register with a 1–2 s batch latency. The sensor hub buffers `CLOCK_BOOTTIME`-stamped events in its FIFO while the processor sleeps, and the phone calls `SensorManager.flush()` before each chirp round and clip fetch, so the timeline is the same. The saving is modest while audio capture wakes the processor every few tens of milliseconds anyway, and larger with AAudio's power-saving path. Batching applies only to sensors the hub can buffer (check `getFifoMaxEventCount()` per model). The step detector already reports at a low rate.

## Orientation

- **Android.** `SensorManager` `TYPE_ROTATION_VECTOR` gives a north-referenced orientation with a heading-accuracy estimate. `TYPE_GAME_ROTATION_VECTOR` uses no magnetometer, so it gives smooth short-term rotation that a power bank or trekking pole can't disturb. `GeomagneticField` gives the declination to true north. Both rotation vectors work in a foreground service.
- **Fused Orientation Provider not used.** Google's Play Services provider is cleaner, but [it delivers updates only while the app is in the foreground](https://android-developers.googleblog.com/2024/03/introducing-fused-orientation-provider-api.html), so it can't serve a locked-screen session. The platform rotation vectors are the single source, which avoids jumps from switching between sources.
- **iOS native.** `CMDeviceMotion` with the `.xTrueNorthZVertical` reference frame, which keeps running in the background audio session.
- **Heading matters little.** Magnetometer heading is often 15–40° off near zippers, power banks or ferrous rock. With 3 or more phones, the layout-to-GPS fit sets the group's rotation, so phone heading matters only for stereo phones and couples. Couples prefer the GNSS walking-track heading over 15–20 m of travel as a soft prior on the inter-phone line (see Couples in the main plan), with magnetometer heading weighted by its reported accuracy.

## Motion

`TYPE_STEP_DETECTOR` and `TYPE_LINEAR_ACCELERATION` (gravity already removed) on Android, and `CMPedometer` and `CMDeviceMotion.userAcceleration` on iOS, give five things:

- the walking/still state that schedules chirp rounds;
- the motion energy between chirp slots, which weights each pairwise clock constraint (a 15 cm shift without a step still puts about 220 µs into a two-way offset);
- the check for movement between a call and the post-detection round, which decides whether that round's geometry may be applied backward to the call;
- the motion energy since the last chirp round, which widens that phone's position uncertainty in the solver (a phone raised with binoculars moves 10–40 cm without a step);
- a per-frame handling score. Grip shifts and footsteps make broadband transients that line up with acceleration spikes, so the phone blanks those few hundred milliseconds before the Stage-1 trigger, and the host down-weights a clip during them.

The trigger is not switched off while walking, because a solo hiker's notifications on the move are a core use. Positions are never computed by integrating acceleration, which drifts by metres within seconds.

## Placement

On Android, `TYPE_PROXIMITY` and `TYPE_LIGHT`, together with gravity from the rotation vector, classify each phone as pocketed, resting or handheld. Each heartbeat carries the result.

- **Pocket or pack:** proximity "near" and ambient light under about 5 lux. The phone flags it at once instead of waiting to miss chirp rounds. The host skips its chirp slot (it couldn't range anyway) and shows the "set your phone down" prompt. The phone stays in readiness queries, where its measured in-band SNR decides whether its clip is used, since a pocketed phone still hears lower-pitched birds.
- **Resting:** flat, with no linear acceleration and no steps for a few seconds. This selects the phone's rigid-surface self-chirp profile measured in Phase 1, since the grip-dependent body-conduction precursor is gone. The roughly +6 dB boundary gain is not added as a weight, because readiness already measures SNR.
- **Handheld:** everything else.

On some Pixels, proximity comes from the under-display light sensor rather than a dedicated part, and screen-off behaviour varies, so Phase 1 checks readings per model with the screen locked. On iOS, proximity monitoring blanks the screen, which would stop the PWA; iOS uses the hearing-quality score and missed rounds only.

## Thermal headroom

Thermal throttling is not only slower inference: a throttled capture thread can stall and drop samples, which breaks sample-count sync. Android's `PowerManager.getThermalHeadroom(forecastSeconds)` forecasts how close the phone is to throttling, and Android 16 adds a headroom listener; iOS has `ProcessInfo.thermalState`. Each heartbeat carries the current headroom or state. The thresholds are config values, starting from:

- headroom above about 0.85 (iOS `.serious`): the phone slows map redraws and leaves classification round-robin for Mic-only;
- above about 0.95 on the host (iOS `.critical`): handover to a cooler phone on online transports (Phase 4); on the hotspot path, the host sheds classification and rendering instead.

A phone sheds load, never capture or chirps.

## Air temperature

Phones have no usable air-temperature sensor. `TYPE_AMBIENT_TEMPERATURE` is almost never present, battery and skin thermistors read 5–15 °C hot during continuous audio and inference, and infrared sensors measure objects. The speed of sound therefore uses weather fetched at session start, corrected for the elevation change since then from the barometer and the standard lapse rate (about −6.5 °C per 1,000 m). The joint solver may refine an effective speed of sound inside a tight bound, as CARDINAL does. Offline with no weather, a seasonal default carries a wider bound. 1 °C is about 0.18% of distance.

## Bluetooth: people nearby

A distant noise source in the woods is often another group of hikers. The acoustic solver already places speech and footstep clusters, and the VAD names speech. A Bluetooth hint can add "people" to a non-bird cluster the classifier can't explain. Strangers' devices can be sensed only through their advertisements' signal strength, because Channel Sounding and UWB need both devices to cooperate. Android only, off by default, with a plain explanation in settings.

- Each phone runs a low-duty scan (`SCAN_MODE_LOW_POWER`) and ignores session members (which advertise the session's service UUID) and the user's own paired devices.
- Every 10–15 s it sends the host only a summary: the number of distinct advertisers above a signal floor and a coarse signal-strength histogram. No addresses, names or payloads leave the phone (addresses rotate every 15 minutes or so anyway).
- The host compares summaries across phones. The phones that hear the most, and loudest, give a coarse sector and range band. Signal strength in forest swings 10–20 dB with bodies and foliage, so no finer claim is made. It tags an unexplained non-bird acoustic cluster in that sector as "people", and may show a subtle "other people nearby · NE" chip. It never draws per-device positions.
- Nothing is kept beyond a rolling 60 s, exported or sent to the cloud. It needs `BLUETOOTH_SCAN` without `neverForLocation` (it derives directions) and location permission, which the app already holds. A privacy review comes before the Phase 5 beta.
- iOS can't scan arbitrary advertisers in the background, and the PWA has no Bluetooth.
- Festival security mode can instead opt into the MinimapPR server's per-device BLE tracking, an operator feature already on the server and never part of birding mode.

## Pressure

`TYPE_PRESSURE` on Android and `CMAltimeter` on iOS give relative heights to roughly 10–20 cm, but phone barometers can disagree by up to about 1 hPa (8 m) in absolute terms. The app locks a phone's offset only when it knows the phone is next to another: at a QR join (the joiner stands next to the host), or when chirp ranging puts two phones under 2 m apart. Either way the offset is good to about the height difference between two people's hands. Online late joiners using a short join code may be far away and at a different elevation, so they wait for the ranging case. Barometers need 1–2 s to settle after the sensor starts, so the offset is taken after that. Weather drift during the session is common to all phones and cancels. GPS altitude is often 15–30 m off and mixes ellipsoid and geoid heights across devices, so it is used only, with a wide uncertainty, for a phone with no barometer.

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
- **Placement-based SNR weighting** (adding the boundary gain of a resting phone): measured in-band SNR already captures it.
- **Placement as a gate or weight on audio** (dropping or down-weighting a phone because it is flagged pocketed or handheld): all placements are expected in real use and the flag is not reliable enough. Measured in-band SNR ranks clips, and motion energy widens position uncertainty.
- **iOS proximity monitoring** for pocket detection: it blanks the screen.
- **Phone temperature sensors** for the speed of sound: see Air temperature.

## Sources

- [Android Developers Blog: Fused Orientation Provider API (foreground-only updates)](https://android-developers.googleblog.com/2024/03/introducing-fused-orientation-provider-api.html)
- [Android reference: MicrophoneInfo](https://developer.android.com/reference/android/media/MicrophoneInfo)
- [Android CDD 5.4, audio recording (`VOICE_RECOGNITION` with noise reduction and AGC off)](https://android.googlesource.com/platform/compatibility/cdd/+/refs/tags/android-mainline-12.0.0_r98/5_multimedia/5_4_audio-recording.md)
