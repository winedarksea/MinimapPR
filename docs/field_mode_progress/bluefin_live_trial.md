# Bluefin live trial (Phase A, 2026-11-22)

Box: 192.168.8.103 (Bluefin/Fedora ostree laptop, user `colin`, `ssh bluefin` via `~/.ssh/config`).
Plan: `make-a-plan-to-unified-cat-of-...md`, Phase A (A1–A7).

## Summary

| Step | Status |
| --- | --- |
| A1 update | Done. `main` @ df95986, `pip install -e ".[birdnet,train]"`, sidecar rebuilt to `dist/`. |
| A2 restart | Done, under `systemd-inhibit --what=idle` + setsid. No orphaned child. |
| A3 baseline | **Blocked: no node traffic.** Neither node has connected since 2026-08-14. |
| A4 trial flags | **Blocked (same cause).** Launcher written and smoke-tested; box left on baseline. |
| A5 BirdNET V3 vs 2.4 | Done, with deviations. Gate condition 4 **fails** (0.45 < 0.8). Default stays 2.4. |
| A6 snippet fixtures | Done. 22 clips registered, `speech_checked = false` until the user listens. |
| B4 sidecar redeploy | Binary from `field-mode-closeout` @ 802079a deployed. Per-trigger measurement pending node traffic. |
| A7 write-up | This file. No systemd unit added; that is a user decision. |

**Read this first: the server's startup housekeeping deleted about 7,300 expired snippets.**
- `data/snippets` went from 9,775 files to 2,476 a few minutes after the first start since 2026-08-14. The
  survivors are 2,467 YAMNet `Explosion` (`security`) clips; every bird snippet is gone.
- They were August clips past their retention expiry (BirdNET-confident clips are kept 30 days). Retention
  ran as configured. No retention setting was changed and no `cleanup` command was run, but starting the
  stack triggered it.
- The A6 export finished at 22:53, before the server came up (~22:56), so the 22 fixture clips are safe.
  Copies: on the box in `~/field_fixture_export/`, and locally in the fixture cache.
- The A5 runs on `data/snippets` loaded their 500 clips into memory at 23:00. Bird audio was effectively
  unavailable for anything later.

## A1. Update

- `git pull --ff-only origin main`: 68728e2 → df95986. The untracked `ingest_smoke.py` was left alone.
- `pip install -e ".[birdnet,train]"` exit 0 (log `~/minimappr-logs/pip_install_closeout.log`).
- `cargo build --release -p minimappr-ingest-sidecar` took 2m47s. The binary was copied to
  `dist/minimappr-ingest-sidecar`. The August binary is backed up at `~/minimappr-ingest-sidecar.bak-68728e2`.
- **Box contention:** an unrelated AutoTS benchmark (`~/AutoTS`, `gift_eval_run_cli --n-jobs 6`, started
  20:18) uses 6 of 12 cores. It was left alone. It inflates the latency numbers below and was competing
  with the stack.

## A2. Restart

- The stack is started via `~/run_minimappr.sh`. That is Mode 2, the Rust sidecar: `DIRECT_INGEST_ENABLED=false`.
  The previous (August) run was a Python split-ingest process, so this is the first sidecar run on the box
  since August.
- **`systemd-inhibit --what=sleep` (and `handle-lid-switch`) is denied over SSH.** Polkit wants
  interactive auth. `--what=idle` is allowed and is what runs. The box is on AC, sits at the gdm greeter,
  and had 13 days of uptime without suspending, so idle is the relevant inhibitor. Lid-close would still
  suspend it.
- New helper scripts on the box:
  - `~/restart_minimappr.sh [launcher]`: kills the inhibit wrapper, the supervisor and the sidecar child
    (no orphan), then relaunches under setsid + `systemd-inhibit --what=idle` and waits for both ports.
    **Run it as its own ssh command.** Its `pkill -f` patterns match any shell whose command line contains
    the sidecar path or `python -m minimappr`, including your own ssh command.
  - `~/run_minimappr_trial.sh`: exports the A4 trial flags, then `exec ~/run_minimappr.sh`.
  - `~/MINIMAPPR_DEPLOY_STATE.txt`: which branch and binary are deployed, and how to return to `main`.
- Processes: `systemd-inhibit` → `python -m minimappr` (:8080) → `dist/minimappr-ingest-sidecar` (:8081).
  The logs are in `~/minimappr-logs/server.log`. The August log was moved to `server.log.pre-closeout-2026-11-22`.

## A3 / A4. Health check and trial flags: blocked, no node traffic

- `GET :8081/api/v1/dsp/status` shows `last_processed_ns: null` and 0 localization attempts across all three
  restarts. Watched from 22:56 to 23:45.
- `/api/v1/nodes` lists `sirith-tetra-07b5` and `sirith-tetra-1a15` as `offline`, last seen 2026-08-14.
- No established inbound connection on :8081. The box's neighbour table has only the router, .199 (the UI
  websocket client) and .131.
- So the nodes are powered off or not on the LAN. `firmware/.../node_config.h` (uncommitted, local) points
  them at `http://192.168.8.103:8081`, which is correct. Fixing this needs someone at the hardware; it is not a server problem.

Baseline vs trial, to fill in once nodes stream:

| Metric (metrics API, 1 h) | Baseline (flags off) | Trial flags |
| --- | --- | --- |
| detections / h | — | — |
| localized fraction | — | — |
| cross-node pair count (`pair_tdoas` across nodes) | — | — |
| `localization_coverage_wait_stop_reasons` | — | — |
| `total_localization_results` / trigger | — | — |

The trial launcher was smoke-tested: started, `/api/v1/config` showed all six flags as intended
(`onset_attribution_loco_enabled=false`), with no errors. The box was then switched back to the baseline
launcher.

**Running now: baseline (`~/run_minimappr.sh`, all flags at defaults) with the B4 sidecar.**

**When nodes are back:**
1. Leave baseline running for 1 h and record the table from `:8081/api/v1/dsp/status`,
   `:8080/api/v1/fusion/status` and the detections API. Do not use log grep: `server.log` is unreliable on this box.
2. Run `ssh bluefin '~/restart_minimappr.sh ~/run_minimappr_trial.sh'`, then record the same 1 h window.
3. Rollback: `ssh bluefin '~/restart_minimappr.sh ~/run_minimappr.sh'`. If cross-node localization is the
   problem, also export `MINIMAPPR_LOCALIZATION_COVERAGE_WAIT_ALL_STREAMING_NODES_ENABLED=false` in the
   launcher.

## B4 sidecar redeploy (requested mid-run)

- Branch `field-mode-closeout` @ 802079a (B1 NaN guard, B4 render-on-deferred-payload, B5 split) was
  moved over with `git bundle` (`~/closeout.bundle`, not pushed) and fetched into a local branch.
- **Deviation:** the build ran in a worktree, `~/MinimapPR-closeout`, with `CARGO_TARGET_DIR=~/MinimapPR/target`.
  `~/MinimapPR` was not checked out to the branch, because the A5 scripts were running from that checkout.
  The Python stays on `main` @ df95986, so only the sidecar changed, as requested.
- `dist/minimappr-ingest-sidecar` comes from 802079a (sha256 prefix `45f3bfbb54682b5d`). The main build is backed up
  at `~/minimappr-ingest-sidecar.main-df95986` (`afea1333493ab574`).
- It restarted cleanly: `worker_running: true`, and `/proc/<pid>/exe` points at the new `dist/` binary.
- **Not measured:** with no triggers there is no `total_localization_results`/trigger and no stop-reason
  histogram. The target is 1.0, down from about 1.03.

## A5. BirdNET V3 vs 2.4 on real site audio

- The models went to `data/models/birdnet_v3/` (FP16 pruned only; no INT8 files exist locally, so the INT8 pairs
  were skipped).
- The regional list was written with `write_v3_regional_species_list` from the persisted site origin:
  `data/birdnet_v3_regional_species_list.txt` on the box, 270 species.
- `birdnet_model_version` is still `2.4`, and `birdnet_v3_species_list_path` was not set in config.

Reports, in `docs/field_mode_progress/bluefin_reports/`:

| File | What |
| --- | --- |
| `birdnet_v3_compare.json` | `compare_birdnet_versions.py` on the 22 exported fixture clips (20 v2.4-labelled birds + 2 ambience). **This is the report used for condition 4.** |
| `birdnet_v3_compare_unfiltered.json` | The plan's literal command (`--audio data/snippets --max-clips 500`). |
| `birdnet_v3_threshold_calibration.json` | `calibrate_birdnet_v3_thresholds.py --seeds 0 1 2 --local-audio bluefin_birds=<fixture clips>`. |
| `birdnet_v3_gate.json` | `birdnet_version_replay_gate.py --seeds 0 1 2` results, with the verdict recomputed against `birdnet_v3_compare.json`. |
| `birdnet_v3_gate_vs_unfiltered_compare.json` | The gate exactly as run (`--bluefin-report` = the unfiltered compare). Same results and verdict. |
| `birdnet_v3_vs_v24_per_clip.json` | Extra: per-clip top-1 for both models (diagnosis below). |

**Deviation, compare input:**
- The literal run took the first 500 files of `data/snippets`, about 4 h of audio. That set is almost all YAMNet noise
  triggers (Explosion / Rain / Silence / Vehicle). It had 1 clip where both models were confident (agreement 0/1), so it is useless for condition 4.
- A bird-labelled 500-clip subset was then built, but housekeeping had already deleted those files.
- So the comparison of record uses the 20 exported bird clips. They are the highest-confidence v2.4 detection per species, so
  **the sample is biased toward v2.4**, and the "labels" are v2.4's own production labels, not ground truth.

**Deviation, calibrate:** the first run failed with `target/release/minimappr-sim missing` (my script built
the sim after calibrate). It was re-run after the sim build, using the fixture clips, since `data/snippets` no longer
has birds.

### Results

Comparison on the bird clips: v2.4 confident on 20 clips, V3 on 11, both on 11. **Mapped top-1 agreement
is 5/11 = 0.455.** Median latency on the contended box: V3 FP16 37.6 ms per audio second, v2.4 28.5 ms.

Per-clip (`v24 top / score` vs `V3 top / score`; "regional" is V3's best class restricted to the 270-species list):

| Detection label (v2.4) | v2.4 | V3 (unfiltered) | V3 regional-best |
| --- | --- | --- | --- |
| trumpeter swan | 1.00 | Trumpeter Swan 0.75 ✓ | same |
| house finch | 0.99 | House Finch 0.90 ✓ | same |
| chimney swift | 0.84 | Chimney Swift 0.98 ✓ | same |
| downy woodpecker | 0.84 | Downy Woodpecker 0.88 ✓ | same |
| house sparrow | 0.90 | House Sparrow 0.85 ✓ | same |
| great horned owl | 0.98 | Great Horned Owl 0.32 (agrees, not confident) | same |
| red-bellied woodpecker | 0.93 | Red-bellied Woodpecker 0.19 (agrees, not confident) | same |
| purple martin | 0.89 | **House Sparrow 0.81** | same |
| red crossbill | 0.87 | **House Sparrow 0.87** | same |
| common nighthawk | 0.84 | **House Sparrow 0.87** | same |
| alder flycatcher | 0.76 | **House Finch 0.80** | same |
| bohemian waxwing | 0.96 | **American Robin 0.81** | same |
| american goldfinch | 0.59 | **Redpoll 0.86** | same |
| eastern screech-owl | 1.00 | Long-winged Conehead 0.28 (out of range) | Great Horned Owl 0.03 |
| northern saw-whet owl | 0.96 | Long-winged Conehead 0.09 | Great Horned Owl 0.03 |
| evening grosbeak | 0.99 | Music 0.17 | Great Horned Owl 0.02 |
| broad-winged hawk | 0.95 | Music 0.19 | White-throated Sparrow 0.04 |
| white-breasted nuthatch | 0.86 | Music 0.33 | Dunlin 0.04 |
| upland sandpiper | 0.81 | Great Horned Owl 0.15 | same |
| black-crowned night-heron | 0.86 | Davis's Southeastern Dog-day Cicada 0.09 | Great Horned Owl 0.04 |

What this shows:
- **The label-swap issue has not cleared; real audio makes it more concrete.** In the sim it was V3 pulling robin, tanager and grosbeak songs to
  Hermit Thrush. Here V3 confidently collapses four different species to House Sparrow, and two others to House
  Finch or American Robin. All of these are common local species, so the regional list does not filter them out: the
  regional-best column equals the unfiltered top in every confident case.
- V3 is not confident on the owls, the night-heron or the grosbeak. It reaches for insects or "Music",
  and its best regional score there is below 0.05. v2.4 calls the screech-owl and saw-whet owl at 0.96–1.00.
  Either V3 is weak on this site's night audio or the August v2.4 labels were wrong. That needs a listen.
  The clips are in the A6 bundle, and the screech-owl/saw-whet species are site regulars (261 and 79 detections).
- Caveat: no ground truth. The clips were picked by v2.4 confidence, which favours v2.4 by construction. A fair
  test needs human-labelled clips, or clips picked by V3 confidence as well.

### Replay gate (`birdnet_version_replay_gate.py --seeds 0 1 2`, about 40 min on the contended box)

The gate ran with `--bluefin-report` pointing at the unfiltered compare, saved as
`birdnet_v3_gate_vs_unfiltered_compare.json`. Its verdict was then recomputed with the script's own
`gate_verdict()` against the bird-clip compare, saved as `birdnet_v3_gate.json`. Both runs give the same verdict:

**FAIL (keep 2.4):** label swaps V3 5 > v2.4 0, and duplicate entities V3 21 > v2.4 20.

`gate_verdict` returns on the sim failures before it reaches condition 4. Evaluated separately, condition 4
fails as well: 0.455 < 0.8.

| Variant | Exact label accuracy | Event recall | Detections per seed | L&C recall at 15 m | Swaps at 15 m | Duplicates at 15 m |
| --- | --- | --- | --- | --- | --- | --- |
| v2.4 | 0.841 (37/44) | 0.917 | 29 / 28 / 27 | 0.19 / 0.38 / 0.38 | **0** | 20 |
| 3.0 | 0.851 (40/47) | 0.979 | 32 / 29 / 28 | 0.19 / 0.19 / 0.19 | 5 | 27 |
| 3.0+regional | 0.851 (40/47) | 0.979 | 26 / 29 / 27 | 0.19 / 0.19 / 0.25 | 5 | 21 |

These match the November 22 run in `birdnet_cutover.md`: V3 has better recall and the same swap blocker.

### Threshold calibration (`birdnet_v3_threshold_calibration.json`)

- 211 clips: 171 sim, 18 fixture, 22 Bluefin.
- The pooled-top-1 quantile-matched V3 thresholds are: trigger/display 0.731 (committed V3 default 0.65),
  routing min 0.724, localize-to-track 0.325 (committed 0.17), log-odds prior 0.206.
- Negative pass fraction is 0 at every matched threshold except the prior (0.037).
- `--write-map` was not passed, so the committed confidence map is unchanged.
- The Bluefin share is small and biased toward v2.4. Do not re-derive the defaults from it.

**BirdNET verdict: keep 2.4.**
- Sim conditions 3 (swaps, duplicates) and real-site condition 4 both fail.
- The label-swap issue has not cleared. On real audio it shows up as V3 confidently collapsing several species to
  common local birds.

## A6. Snippet fixtures

- 20 bird clips (one per species, highest `label_confidence`, categories `wildlife`/`unknown` with a BirdNET
  species label, coyote and drone excluded) and 2 ambience clips (most recent `Wind` and `Rain`) were exported
  read-only from `data/minimappr.db`. Bundle: `bluefin_bird_snippets_v1.tar.gz` (11.4 MB, sha256 `5cb0da6f…`). It
  stays on the box at `~/field_fixture_export/`.
- It was registered with `scripts/fetch_field_fixtures.py register --extract`. The `[[bundle]]` entry replaced the
  commented example in `tests/fixtures/field_bundles.toml`.
  - The source is `ssh://colin@bluefin:22/...`, where `bluefin` is the `~/.ssh/config` alias that supplies the key.
    The raw IP failed to authenticate through the script's rsync.
  - `fetch` verified size and sha256 and extracted the bundle to
    `~/.cache/minimappr/field_fixtures/bluefin_bird_snippets_v1/bluefin_bird_snippets_v1/`.
  - `tests/test_field_audio_fixtures.py`: 49 passed.
- **`speech_checked = false`.** Every clip is 30 s or shorter of private site audio, and a person must listen to each one
  before it can be set to true. Drop any clip with speech, then re-tar and re-register. The bundle stays out of git.

## A7. Open items

- **Nodes are offline.** A3/A4 and the B4 per-trigger measurement all need streaming nodes.
- **Systemd user unit (user decision, not added).** The box still has no autostart. A user unit with
  `loginctl enable-linger colin` would survive reboots. It would not by itself stop lid or sleep suspend; that needs
  `sudo systemctl mask sleep.target suspend.target` or a logind `HandleLidSwitch=ignore` drop-in, either of which
  needs sudo on the box.
- The BirdNET default stays 2.4: gate condition 4 failed and the label-swap issue still stands. A fair re-test
  needs labelled site clips.
- The snippet deletion above. If more real-site audio is wanted for BirdNET work, export it before any
  restart that runs housekeeping, or raise the retention for a capture window.
- Return the box to `main` after merge: see `~/MINIMAPPR_DEPLOY_STATE.txt` or the closeout report.

## 2026-11-23 follow-up: A3 baseline and B4 check with live nodes

Setup: indoors, `sirith-tetra-07b5` (192.168.8.126 or .188) and `sirith-tetra-1a15` about 4 m apart, the
speaker playing native birdsong roughly between them, and no GPS fix. `1a15` joined partway through the
window. Baseline launcher (`~/run_minimappr.sh`, all flags at defaults), sidecar at 802079a.

| Metric (14:49–15:50 UTC, 1.01 h) | Baseline (flags off) |
| --- | --- |
| Detections emitted | 67, all omni from `07b5`; 0 localized |
| Bird detections | 2 (eastern phoebe 0.76, evening grosbeak 0.40) |
| Other labels | Explosion 25, Crackle 21, Rain 9, Sound effect 6 (room tone at -62 to -69 dB) |
| Localization attempts / results | 4,124 / 4,124 |
| Coverage wait stop reasons | all_covered 4,431, stalled 109 (max 999 ms), superseded 1 |
| Sidecar failures (dropped frames) | 57, all after `1a15` joined; cause not visible because `server.log` stops growing |
| Tracks | 0 |

- **B4: passes.** Exactly one `localization_result` per trigger (1.0, down from about 1.03).
- **The speaker was too quiet.** A render's spectrum was flat from 50 Hz to 16 kHz at about -42 dBFS.
  BirdNET labelled only 2 calls in an hour of continuous playback.
- **Both nodes report the same position** (-1.8, 2.97, 0). Operator pins through `PATCH /api/v1/nodes`
  are stored but have no effect in Rust ingest (TODO.md). So nothing between the two nodes localizes.
- After the window the sidecar was rebuilt at 26d13e0 (F1 never-streamed-member render fix) and
  installed with `cp --remove-destination`. The previous binary is `~/minimappr-ingest-sidecar.closeout-802079a`.
- A4 (trial flags) not yet run.
