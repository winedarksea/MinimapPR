# Faint-call admission: both ingest backends, two candidates

Review item F (faint-call admission). This is a measurement only: every default is unchanged. The two candidates sit behind new settings that default to off.

## Recommendation

| setting | today | recommended | why |
|---|---|---|---|
| `trigger_noise_floor_relative_enabled` (direct ingest) | off | **on** | Quiet site: admission goes from 0% to 69% of calls with no false triggers. In rain it costs the same 77/h steady false triggers that the sidecar already runs with by default. This also gives both backends the same semantics. |
| `trigger_short_block_excess_enabled` / `localization_trigger_short_block_excess_enabled` | off | **on, after a warm-up fix** | Sidecar, quiet site: admitted rises from 50% to 73% and solved from 46% to 65%. Chips admitted go from 1/8 to 6/8, and straddling chips from 0/4 to 3/4. Steady false triggers rise only 77 to 83/h in rain. The catch is startup: the block fluctuation term starts at 0 dB and converges on a 20 s time constant, so rain produces a burst (19 vs 6 false triggers in the first 30 s). Seed or warm up `mean_short_block_excess_db` before turning it on. |
| `omni_scan_clearest_channel_gate_enabled` | off | **off** | No call was classified in either mode at any scan phase. The binding limit is the absolute `omni_scan_min_rms = 0.001` over a 21 s window, not the channel mean. The clearest channel is only about 3 dB above the mean. A gate relative to the floor is the follow-up worth measuring. |

Compute cost of the short-block candidate is about 1.5x the trigger decision. Python: 22 to 34 µs per 4 x 960 frame. Rust: 4.0 to 8.5 µs (`cargo test --release -p minimappr-core short_block_cost -- --ignored --nocapture`, measured at load average ~20). Either way it is negligible beside the sidecar's 350-500 CPU s per 2-minute run.

## Results

Run: `MINIMAPPR_ENABLE_FAINT_ADMISSION_SWEEP=1 pytest tests/test_faint_call_admission_sweep.py::test_faint_call_admission_sweep`. It took 17 min 47 s wall on 2026-11-26, with 1-minute load average 2.4-10.6 from unrelated processes, recorded per cell. Each cell has 26 calls: steps of 16, 20, 24 and 28 dB in-band SNR per kind and alignment, plus two startup syllables.

| candidate | background | admitted | solved | classified | stored localized | false trig/h (app, 45 s exposure) | false trig/h (offline, 20 min steady) | offline startup false trig (first 30 s) | trigger µs/frame (Python) | sidecar CPU s |
|---|---|---|---|---|---|---|---|---|---|---|
| sidecar_default | quiet | 50% | 46% | 54% | 54% | 0 | 0 | 0 | 23 | 478 |
| sidecar_short_block | quiet | **73%** | **65%** | 62% | 62% | 0 | 0 | 0 | 34 | 515 |
| direct_default | quiet | 0% | 0% | 0% | 0% | 0 | 0 | 0 | 22 | - |
| direct_relative | quiet | 69% | 12% | 69% | 69% | 0 | 0 | 0 | 23 | - |
| direct_relative_short_block | quiet | 73% | 12% | 73% | 73% | 0 | 0 | 0 | 34 | - |
| sidecar_default | rain | 46% | 38% | 50% | 50% | 0 | 77 | 6 | 23 | 392 |
| sidecar_short_block | rain | 54% | 42% | 42% | 42% | 322 (3 of 4 before 30 s) | 83 | 19 | 34 | 351 |
| direct_default | rain | 46% | 0% | 46% | 46% | 81 | 0 | 0 | 22 | - |
| direct_relative | rain | 65% | 4% | 65% | 65% | 161 | 77 | 6 | 23 | - |
| direct_relative_short_block | rain | 65% | 4% | 69% | 69% | 564 (5 of 7 before 30 s) | 83 | 19 | 34 | - |

Every upload was received in every cell. The sidecar drained within 3-11 s of the last upload in every cell.

What the per-kind breakdown shows (full table in the sweep's markdown output):

- **Chips (12 ms, shorter than a 20 ms frame) are what the frame-RMS trigger loses.** On the sidecar's default, the faintest chip admitted is 28 dB and a chip straddling a frame or upload boundary is never admitted. With short blocks, both alignments are admitted from 20 dB at the quiet site.
- Syllables and sustained calls are admitted from about 20 dB on every relative-trigger configuration. Short blocks neither help nor hurt them.
- Direct ingest with its relative trigger on admits about as much as the sidecar with short blocks. That is because its trigger sees each 20 ms frame, while the sidecar's absolute gate looks at the 80 ms window ending at each upload.
- In rain, the direct default's absolute 0.001 gate admits loud calls (46%) because the rain floor is about 4x higher. At the quiet site it admits nothing.

## Not about admission, but visible here

- **Direct-ingest solves are mostly wrong under E7's settings.** Solved is 4-12% against 65-73% admitted. Bearing errors run 10-126° on stored "localized" detections. E7's `MINIMAPPR_LOCALIZATION_SRP_GRID_RESOLUTION_M=0.5` / `SEARCH_PADDING_M=2.0` suit the sidecar; the Python SRP grid search likely clips a source 30 m out to the grid edge. Not investigated further.
- The sidecar renders the classification window about once per second with no energy gate (`birdnet_omni_fallback`). Its omni coverage is continuous, unlike the Python scanner's 21 s window every 30 s behind an absolute gate. With a constant classifier these renders say nothing about recall, so they are reported as `periodic omni coverage` and left out of the stages.
- Sidecar event triggers emit several manifests sharing one window centre. The sweep groups them, so counting manifests overstates triggers.

## Method

- **Fixtures** (`tests/faint_admission_sweep_fixtures.py`): a single tetra node from E7, with the source 30 m away.
  - Calls: a 12 ms FM chip, a 60 ms FM syllable and a 1 s vibrato whistle. Each is placed inside one frame and centred on an upload boundary, at 4 steps. Two syllables land inside and just past the 1 s warm-up. A 20 s noise-only tail follows.
  - Backgrounds: quiet is MEMS self-noise at 2.5e-5 RMS. Rain is a 1e-4 floor plus about 150 raindrop clicks/s per microphone, independent per microphone, 2.2e-4 RMS in total.
  - The SNR is in-band (400-3200 Hz) at the nearest microphone, over the background's own in-band RMS.
  - All of this is synthetic: the repo holds no held-out field recordings.
- **Backends**: `run_streaming_scene_through_full_app(ingest_backend=...)` streams byte-identical firmware uploads either to the real sidecar or to the app's direct `/api/v1/ingest/binary`. Direct ingest uses `tests/realistic_scene_direct_ingest.py`. Both use a constant bird classifier.
  - Sidecar manifests are followed live (`tests/sidecar_manifest_collector.py`), because the 512-event replay ring overflows on a 2-minute run.
  - The harness waits until the sidecar has solved up to the end of the capture. Without that wait, a loaded machine left the last ~30 s unscored in a first run.
- **Stages** (`tests/faint_admission_sweep.py`):
  - received: every covering upload was accepted.
  - admitted: an event trigger fired within one localization window of the call.
  - solved: bearing within the tier scorer's 10°. For the sidecar this comes from the solve manifest. For direct ingest it comes from the stored localized detection, the only solve that path exposes.
  - classified: a trigger-driven detection stamped within the call carries the label.
  - stored: that detection is localized.
- **False triggers**: triggers outside every call's window (0.5 s before to 2 s after) per hour of that exposure. The app runs give only 45 s of exposure (1 trigger = 80/h). The resolution figure is therefore an offline replay of the Python trigger decision over 20 min of each background (`tests/faint_admission_offline_replays.py`). Its relative-trigger decisions equal the Rust port's (parity goldens). It does not model the sidecar's absolute gate, which uses 80 ms windows instead of frames.
- **Omni**: `ContinuousOmniScanner._scan_node` replayed offline on the sweep capture at the scanner's startup phase and three more phases. The full-app runs keep it off because it schedules on wall time while the harness streams at 2x.
- **Reduced for run time**: 4 SNR steps instead of 5, a 20 s noise tail instead of 30 s, and one node. Whole sweep under 18 min at light load; the first run at load 15-25 took 23 min.

## Candidate semantics (same in Python and Rust)

Short-block excess (`ingest_relative_trigger.py`, `minimappr-core/src/stage1/noise_floor_relative_trigger.rs`):

- The frame's loudest 5 ms block on any channel is measured. Blocks tile from sample 0, and one extra block ends at the last sample. That level is judged against the node's floor, which still follows the whole-frame level.
- The bar is `snr_db` plus 1.5 x its own slow mean excess of non-triggering frames.
- The result is OR'd with today's frame decision, so every frame that triggers today still does.
- It only acts with the relative trigger on. The parity goldens in `tests/test_attribution_trigger_cross_language_parity.py` (`relative_trigger_short_block`, `peak_short_block_rms`) hold both implementations to the same decisions.

Clearest-channel omni gate (`omni_scanner.py`): the scanner gates and classifies the highest-RMS channel instead of the channel mean.
