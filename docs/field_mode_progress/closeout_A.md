# Closeout Phase A: Bluefin ops

Details: `bluefin_live_trial.md`. Reports: `bluefin_reports/`.

| Step | Status | Notes |
| --- | --- | --- |
| A1 | Done | Box on `main` df95986. pip extras installed. Sidecar rebuilt; August binary backed up. |
| A2 | Done | `systemd-inhibit --what=idle` + setsid via `~/restart_minimappr.sh`. Sleep/lid inhibit is refused over SSH (polkit). No orphan. |
| A3 | Blocked | No node traffic. Both nodes offline since 2026-08-14 and not on the LAN. |
| A4 | Blocked | Same cause. `~/run_minimappr_trial.sh` written and smoke-tested (flags confirmed via `/api/v1/config`). Box left on baseline. |
| A5 | Done (deviations) | Gate FAIL (V3 swaps 5 > 0, duplicates 21 > 20); condition 4 also fails, 0.455 on 11 both-confident clips. Label swaps still present on real audio (V3 collapses 4 species to House Sparrow). Default stays 2.4. |
| A6 | Done | `bluefin_bird_snippets_v1` (20 birds + 2 ambience) registered, `speech_checked = false`, fixture tests 49 passed. |
| B4 redeploy | Deployed, unmeasured | Sidecar from `field-mode-closeout` 802079a via git bundle + worktree. Per-trigger count needs node traffic. |
| A7 | Written | Systemd unit not added; question for the user. |

## Deviations

- **Snippet loss (not caused by a config change, but caused by starting the stack).** The first server
  start since August ran retention housekeeping and deleted about 7,300 expired snippets (9,775 → 2,476 files;
  no bird clips remain). The A6 export had finished beforehand.
- **A5 compare input.** The literal `--audio data/snippets --max-clips 500` sampled noise triggers, giving 1
  both-confident clip. Kept as `birdnet_v3_compare_unfiltered.json`. The comparison of record is on the 20 exported
  bird clips, which are biased toward v2.4.
- **A5 calibrate.** Re-run after the sim build, with `--local-audio` set to the fixture clips.
- **B4 redeploy.** Built in the worktree `~/MinimapPR-closeout` instead of checking out `~/MinimapPR`, so the
  running A5 scripts and the Python stay on `main`.
- **systemd-inhibit:** `--what=idle` only.

## Open

- Power the nodes up and get them on the LAN, then run A3 → A4 → the B4 per-trigger check (procedure in `bluefin_live_trial.md`).
- User listen-check of all 22 fixture clips.
- Systemd unit decision.
- Return the box to `main` after merge (`~/MINIMAPPR_DEPLOY_STATE.txt`).
