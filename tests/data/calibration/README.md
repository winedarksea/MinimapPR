# Calibration replay bundles

Drop `calibration_{session_id}.zip` bundles exported from a MinimapPR server
(`GET /api/v1/calibration/{session_id}/bundle`) into this directory.
`tests/test_calibration_replay.py` discovers every `*.zip` here, replays its
raw multi-node audio through the full localization + classification pipeline,
and enforces the pass thresholds from the bundle's `expectations.json`
(falling back to `minimappr.sim.replay.DEFAULT_EXPECTATIONS` when absent).

Bundles are gitignored (`.gitignore` re-includes only this README); tests
skip when no bundles are present.

Bundle schema: see `minimappr/calibration/bundle.py` — manifest.json,
ground_truth.json, detections.json, optional expectations.json,
audio/{node_id}.wav, and optional `reference_audio/`. Ground truth v2
(individual_id, species lists, echo sources, trajectories) and the entity
expectations are specified in `docs/field_mode_progress/bundle_format_v2.md`.
