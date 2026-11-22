# W3-mesh-sim: Layer 2 deterministic mesh simulation

Branch `fm/mesh-sim` (off `field-mode` at "Merge fm/core-sync into field-mode"). Row "W3-mesh-sim" in `build_out_plan.md`; test strategy §Layer 2.

## What exists

### Sans-IO state machines: `minimappr-core/src/mesh/`

Every machine takes `(now_ms, input)` plus an injected capability and returns actions. There is no tokio, no thread, no wall clock and no randomness. `now_ms` is the coarse OS session clock and drives timers only. Anything TDOA-related is a sample index on a node's own clock, mapped through island clock models. `lib.rs` gains one line, `pub mod mesh;`.

| Module | Role |
|---|---|
| `node_state_machine`, `node_message_handling`, `node_host_link` | Node role. Covers join, heartbeats, the detection outbox, chirp slots and arrival reports, readiness answers, clips, mirroring the event log with `since=N` catch-up, solo fallback and rejoin, takeover, Mic-only on low thermal headroom, and classification duty from the rota. The shell implements `NodeAudioWindowSource`: readiness of a window and clip encoding from the rolling buffer. |
| `host_state_machine`, `host_message_handling` | Host role. Covers the transport policy, membership, the event log and its broadcast and catch-up, chirp rounds, the call pipeline, entity updates, the rota, and thermal relief: handover when online, classification shedding on the hotspot. It also resolves rival hosts (higher term wins; on a tie, earlier join order wins). |
| `event_log` | `SessionEventLog` (host) and `MirroredEventLog` (node). Entries hold encoded `FieldMessage` frames, so an older mirror keeps kinds it does not know. Detections are logged at most once per `(node_id, local_seq)`. A gap triggers `since=head` catch-up. A newer term rewrites a conflicting tail. |
| `detection_outbox` | A detection leaves the node's queue only when it shows up in the node's own mirror; the log echo is the acknowledgement. One mechanism therefore covers live sending, lost messages, host death and solo fallback: everything ends as an idempotent re-upload. |
| `host_membership`, `host_restoration` | Join order (the succession order; a PWA is never eligible), presence, and heartbeat facts. A successor rebuilds from its mirror: the log, membership, entity ids, transport and term. |
| `host_sync_pipeline`, `island_sync_engine`, `island_round_solver`, `layout_frames`, `chirp_scheduler` | Chirp rounds feed the `sync` and `layout` modules unchanged: BeepBeep pairs with closing-chirp rate compensation, then a two-pass clock graph, `PhoneClockFilter`, acoustic islands, `solve_layout` and `fit_layout_to_gps`. Each island has its own clock epoch, filters, layout and anchor. Scheduling is silent-until-needed (heartbeat, post-detection, group stopped, clock invalid, new member, fresh domain) or periodic. |
| `chirality_cue` | Locks layout chirality from two bearings that one stereo phone or array node takes in the same round. The signed angle between the two bearings does not depend on rotation, so the phone's heading error cancels. |
| `event_matcher`, `readiness_selection`, `clip_budget`, `call_pipeline`, `call_pipeline_types` | Detection → call. A call is opened on one detection above the localize-to-track score, or merged into an existing call's window. The host sends a readiness query to the classifier's clock domain, re-sending unanswered queries once. Clip selection keeps the classifier first and fills up to the cap by in-band level plus a spread bonus; nodes left out by the cap are counted. Fetches go through a per-island token bucket. A clip is discarded (and counted) if its node's clock epoch changed since the request. Late clips are abandoned, not retried. The solver sits behind the `ClipSetLocalizer` seam. |
| `srp_seed_refiner` | **`TdoaSeedRefiner` for `srp_layout::LmSeed`.** LM over whitened pair TDOAs, where σ combines timing with both nodes' position σ. A weak height prior, worst-first outlier rejection, and an `(JᵀJ)⁻¹` covariance that refuses unobservable geometry. The horizontal covariance is then inflated by the fix's drift across the seed's height band, because on a flat phone layout height is barely observable. |
| `arrival_time_localizer` | Arrival times → a Cauchy polar-grid seed → `JointLmSeedRefiner`. Pairs are formed against the node with the best combined timing and position σ. If χ²/dof > 9, it retries leaving each node out in turn, which catches a stale clock on the reference node. Usable for a clap test. The simulator uses it in place of clip SRP. |
| `host_transport_policy`, `classification_rota`, `clock_mapping`, `local_tangent_plane`, `config` | Transport modes Auto / Local / Online. A constrained link is treated as no signal. Rota clumps are single-linkage groups within 3 m; duty is assigned by time slot on the island clock. The wire-form clock map. A flat east-north plane. Every tunable, grouped by concern. |

**Protocol (additive).** New kinds 25–33 in `protocol/mesh_messages.rs`:

- `EventLogAppend` and `EventLogCatchUpRequest`
- `HostAnnouncement` and `HostHandover`
- `SessionMemberRecord`
- `NodeStatusChange`
- `ChirpDirectionReport`
- `ClassificationRota`
- `TransportNotice`

New u8 wire enums are `NodeStatusChangeCode`, `SessionTransportCode` and `SessionChangeReasonCode`. `NodeId` derives `Default`. `tests/protocol_messages.rs` covers every new kind.

### Simulator: new crate `minimappr-mesh-sim`

It depends on `minimappr-core` and on `minimappr-sim` (physical constants, ISO 9613-1 absorption). Neither of those depends on it. It is a root workspace member.

| Module | Role |
|---|---|
| `seeded_random` | The `MINIMAPPR_SIM_SEED` override, the 10 `PULL_REQUEST_SEEDS`, per-purpose ChaCha streams, and pure-hash Gaussians. A draw never depends on the order in which state machines ask. |
| `acoustic_world`, `device_spec` | **The arrival-time abstraction.** True positions and trajectories. Clocks with θ ±20 ms and α ±60 ppm, minus samples lost in capture stalls. Call and chirp arrivals on each device's own clock. SNR from 1/r spreading, ISO absorption and ambient noise, with thresholds for Stage-1, the classifier and clip correlation. Timing noise scales as 1/√SNR. Also GPS bias and jitter, thermal ramps with stalls (alternately reported and silent), speech and handling intervals, muted speakers, and stereo chirp bearings (common heading error plus per-bearing noise). |
| `simulated_network` | The relay needs a cell link at both ends and enforces a message cap. The hotspot is range-limited around the host and joined after a delay. Latency jitter reorders messages; loss is Bernoulli. Bulk clips serialize on the uplink. Scripted per-device drops, disconnects and extra clip delay. Every byte is counted by kind and by whether it crossed a constrained link. Frames really go through postcard encode and decode. |
| `scenario_runner`, `runner_actions`, `device_harness` | Discrete-event loop on a virtual clock. The device harness is the "shell": it drives `FieldNodeStateMachine` and, on the hosting device, `FieldHostStateMachine`. It implements `NodeAudioWindowSource`, carries out takeover with `take_over(mirror)`, transport switches and step-down. |
| `synthetic_clip` | Clips carry arrivals with noise, padded to real FLAC size (0.57 B per device sample). `ArrivalClipLocalizer` implements `ClipSetLocalizer` and records traces. |
| `scenario_outcome`, `invariants` | Fixes are scored after a proper rigid alignment of the host's node positions to the true positions. An anchor offset is therefore not counted as error, while a mirrored layout is. Common invariants: no duplicate detections, no silent loss, every mirror an exact prefix that converges within 3 s, clips ≤ cap per call and sent ≤ requested, no TDOA across clock domains, no stale clip, no confident mirror, fixes inside 4σ + 0.5 m (a flagged low-confidence mirror is exempt). |
| `scenarios/*` | 16 scenarios (below). |
| `tests/pr_seeds.rs` | One test per scenario × 10 fixed seeds, or the single seed in `MINIMAPPR_SIM_SEED`. |
| `examples/nightly_mesh_seeds.rs`, `scripts/nightly_mesh_sim.sh` | Nightly runner (`--seeds N --start-seed S --scenario NAME --threads T`), multi-threaded. Prints each failing seed with its reproduce command; exit code 1 on any failure. |
| `examples/inspect_scenario.rs` | Dumps records, traces, islands and link state for one failing seed. |

## Scenarios

All scenarios use the arrival-time abstraction. Layer 1 audio was not needed: what these scenarios test is scheduling, matching, sync bookkeeping and network behaviour. Waveform effects are Layer 1's, and the core sync and SRP streams already test them on rendered audio. Arrivals are still fed through the real ranging, clock graph, filters, islands, MDS and GPS anchor code.

| Scenario | Setup | Key assertions (beyond the common invariants) | PR seeds |
|---|---|---|---|
| baseline_localization | 5 phones, 3 calls | 3/3 fixes, < 3 m gate, inside region, no loss, convergence | 10/10 |
| host_death | Host killed mid round, successor killed mid clip fetch | Succession follows join order, one live host, successor's first round starts ≤ 1 s and finishes ≤ 6 s after takeover, post-takeover calls localized | 10/10 |
| network_faults | 10 % loss, one phone's clips +800 ms, a joiner disconnects mid call | ≥ 2/5 localized, the rest abandoned cleanly, no stale clip, logs converge after loss stops | 10/10 |
| route_change | Input route change while the triggered-round allowance is spent | Invalidated phone not fetched before its resync round, contributes after it | 10/10 |
| island_split_merge | 8 phones → 4 + 4 at 150 m → back | Split into two domains, island-local fixes only, merge within one heartbeat with a fresh epoch | 10/10 |
| offline_host_loss | Hotspot session, 2 phones out of range ~60 s; 30 % loss on return | Solo then rejoin, solo detections logged once, resynced | 10/10 |
| walking_at_call | Group walks, bird calls, stop 2 s later | Walking phones use the pre-call layout (the still phone does not), fix inside its region | 10/10 |
| mirrored_layout | 4 phones, r = 5 m, GPS σ 6 m, stereo cue at 35 s | No confident mirror before the cue, none at all after the lock | 10/10 |
| readiness_cap | 8 ready phones, cap 6 | Exactly 6 fetched, classifier included, 2 counted | 10/10 |
| mic_only_round_robin | 2 Mic-only phones + a 3-phone clump | Mic-only phones never classify but contribute clips; clump classifies 1.00× per call | 10/10 |
| muted_phone | One silent speaker | Banner by the end of its first round, flag logged, 0 muted ranges, own domain, others localize | 10/10 |
| transport_fallback | Relay drops 2 of 5 for 25 s, then cap; separately host loses cell, one phone out of hotspot range | Solo + catch-up, `TransportNotice` with reason, no duplicates, no loss | 10/10 |
| thermal_shedding | 10-minute compressed session (30 min in the strategy), hot phone + hot host | Mic-only before first stall, online handover before host stall, hotspot sheds instead, hot phone resynced after stalls | 10/10 |
| earbuds | Android earbuds (pinned), native iOS earbuds | Android keeps slot and clock; iOS loses slot, keeps clock, told it is listen-only | 10/10 |
| constrained_link | Host cell link turns constrained | Hotspot started, logged, 0 clip bytes over the constrained link | 10/10 |
| faint_bird_one_classifier | Bird 10 m from one phone, 32–39 m from three, 70 dB SPL | Stage-1 fires on the near phone only, all 4 clips fetched, fixes < 3 m | 10/10 |

## Commands, counts, timings (Apple M2, 2026-11-22)

- `cargo test -p minimappr-mesh-sim --test pr_seeds`: 17 tests (16 scenarios × 10 seeds plus a catalogue check) in **0.54 s** wall time (parallel), **1.48 s** with `--test-threads=1`.
- `cargo test -p minimappr-core --lib mesh`: 21 unit tests, including the HostLink regressions and the 3-node and stale-reference localizer cases.
- `scripts/nightly_mesh_sim.sh 200 1000` (release): 3 200 runs in **4.8 s**, **5 failed (0.16 %)**. With start seed 50 000: 7 failed (0.22 %). Failures are listed under open issues.
- `cargo test --workspace`: green. Counts: core lib 192 (21 of them in `mesh`), core integration 7+9+8+9+4+1+2+2+9+7, sidecar 116, mesh-sim 17, sim 25.
- `cargo build -p minimappr-core --target wasm32-unknown-unknown` and `--target aarch64-linux-android`: both pass.
- `cargo clippy -p minimappr-core -p minimappr-mesh-sim --all-targets`: no warnings in `src/mesh`, `src/protocol` or the mesh-sim crate.

The root `Cargo.toml` gains `[profile.test.package.minimappr-core] opt-level = 3`, plus `opt-level = 3` for `minimappr-mesh-sim` (test and dev). Unoptimized, the solver work alone took 0.45 s per run. This also speeds up core's slow debug tests (codec stream note: `stage1_trigger` took about 16 s).

## Bugs the simulator found (all fixed here)

In the new state machines:

1. **Reconnect hijack.** A phone walking back into range saw a long host silence and took over at once; the old host then yielded to the higher term. Silence is now counted from link-up. Takeover happens only on the relay, because on the hotspot the link dies with the host. Regression test in `node_host_link`.
2. **Equal-term rivals never converged.** A node rejected a same-term announcement from another host. Join order now breaks the tie.
3. **Carry-forward across MDS frames.** Unranged nodes kept their old local coordinates inside a new solve's arbitrary frame, which placed them metres off. They are now moved with a rigid map (reflection allowed) over the common nodes.
4. **Flip ambiguity.** A node with < 3 ranges got a confident but mirrored position. Such nodes are now skipped, and a non-rigid layout is rejected.
5. **Walking phones in a round.** Offsets corrupted by motion (milliseconds at 3 m/s) initialized clock filters, and the next good round then looked like a silent gap. Walking phones are now graph vertices only: they get neither an offset nor a position from that round.
6. **Silent stall inside a round.** It biased every range of that node by about c·stall/2. The fix has two parts:
   - An offset jump beyond 6σ and 1 ms re-solves the layout without that node and re-seeds its filter in the same round.
   - A node with ≥ 2 rejected ranges is dropped from that round's layout.
7. **A lost schedule or report split an island.** "Not heard" now counts only for emitters that provably chirped audibly. A fresh clock domain (after a merge or split) is surveyed at once.
8. **Joiners never surveyed.** New, returning and lone nodes are provisional: they join every round, and a not-yet-surveyed one triggers a round.
9. **Single-bearing chirality.** With GPS σ of 6 m the anchor rotation is uncertain by tens of degrees, so single bearings flip the lock (see core findings). Replaced with bearing pairs.
10. **Overconfident covariance.** The reported regions were too small in four ways:
    - the arrival localizer's reference node shared position error with every pair;
    - unknown phone heights were missing (σ now 0.25 m);
    - the linearization near the phones' plane hides the height/horizontal coupling (the height-band drift term now covers it);
    - leave-one-out accepted under-determined subsets.
11. Readiness answers lost under 10 % loss silently dropped nodes from calls. Unanswered queries are now re-sent once.
12. A successor host reused call ids. Each term now starts at `term << 32`.

## Findings about earlier core modules (not changed)

- `layout::gps_anchor_fit::chirality_from_bearing` judges a single world bearing against the GPS-fitted rotation. With GPS σ of 6 m on a layout about 10 m across, that rotation is often 20–40° off, and decisive-looking wrong locks were common: 6 of 200 seeds locked wrong. It is correct as written; it just needs either a separate rotation reference or a bearing pair. `mesh::chirality_cue` uses pairs.
- `fit_layout_to_gps` chirality confidence is occasionally confidently wrong on small layouts: 0.997–0.99998 on about 4 of 3 200 nightly runs (split clumps of 4, walking group). Possibly a calibration issue. In the walking case the mesh mixes positions from two layouts, each with its own anchor, which may cause it instead. Not proven to be a core bug.
- `robust_layout_refine::LayoutSolution::is_rigid` is infinitesimal rigidity. A node with two ranges passes it and can still be flipped. This is usage guidance, not a bug.
- `PhoneClockFilter` has no way to seed from a published `ClockModel`, so a successor host re-surveys (one round) rather than restoring clocks.

## Deviations

- **No Layer 1 audio in any scenario.** Arrival-time abstraction throughout (see above). A clip-SRP `ClipSetLocalizer` over real FLAC clips (`srp_layout` + `JointLmSeedRefiner`) is the production implementation still to write. The seam and the refiner exist.
- **island_split_merge uses 4 + 4 phones at 150 m**, not 3 + 3 at 70 m:
  - A 3-phone island has two TDOAs; with unknown height its fix is often an honest `Singular`/`IllConditioned` failure.
  - This world's omnidirectional 85 dB chirps still close two-way at about 90 m.
- **thermal_shedding** compresses 30 minutes to 10 and uses 30 s periodic rounds. The hot phone cools after shedding (a ramp with recovery), so it can be checked for resync.
- **Hotspot host death leads to solo, not takeover.** On the hotspot a node cannot tell a dead host from its own isolation, and an automatic second hotspot is the offline federation the plan rules out. Takeover from the mirror is exercised on the relay.
- **The successor re-surveys rather than restoring clocks and layouts**, and does not re-run calls that were pending at the dead host. Their detections are in the log; the clips were lost with the host's in-memory state.
- **Not built:** "Dawn chorus", "Mixed topology", "iOS PWA node", "People nearby", couples and collinear trails, which are W4 / server-track ports or need DOA nodes. The relay transport and chunking belong to W3-relay. The sim models the transport abstractly.

## Open issues

- Nightly residue, about 0.2 % of runs. Each prints its reproduce command, and after a fix it joins `PULL_REQUEST_SEEDS`:
  - confident GPS-chirality mirrors (split clumps, walking group; see findings);
  - region misses after a wanderer rejoins or a host stall (offline_host_loss 50005/50017/1185, thermal_shedding 1195/50011);
  - gate misses by 0.1–0.2 m (faint_bird 1131, baseline 50187).
- In walking_at_call, mixing pre-call and post-call layouts in one solve relies on both being GPS-anchored. Without GPS, frames would mix.
- Region coverage is checked at 4σ + 0.5 m per fix. The strategy's 90–98 % contour calibration needs a statistical gate (W4-sim-gates).
- Readiness answers carry no timing quality, and the cap's spread bonus is a heuristic. Both need tuning on golden bundles.
