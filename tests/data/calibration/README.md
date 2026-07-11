# Calibration replay bundles

Drop `calibration_{session_id}.zip` bundles exported from a MinimapPR server
(`GET /api/v1/calibration/{session_id}/bundle`) into this directory.
`tests/test_calibration_replay.py` discovers every `*.zip` here, replays its
raw multi-node audio through the full localization + classification pipeline,
and enforces the pass thresholds from the bundle's `expectations.json`
(falling back to `minimappr.sim.replay.DEFAULT_EXPECTATIONS` when absent).

Bundles are gitignored (`.gitignore` re-includes only this README); tests
skip when no bundles are present.

Bundle schema (v1): see `minimappr/calibration/bundle.py` — manifest.json,
ground_truth.json, detections.json, optional expectations.json,
audio/{node_id}.wav, and a reserved empty `reference_audio/` directory.
