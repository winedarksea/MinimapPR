# Classifier routing

Classification is driven by `data/classifier_routing.json` (schema `version: 1`),
loaded by `minimappr/classifiers/routing.py`. It replaces the old
`classifier_backend` setting and `data/model_chain.json` file (removed — startup
raises with a migration message if either is still present/set).

## Shape

```json
{
  "version": 1,
  "classifiers": { "<member_id>": { "backend": "...", ...backend kwargs } },
  "contexts": { "<context_name>": { "run": ["<member_id>", ...] } },
  "chains": [ { "id": "<member_id>", "after": "<member_id>", "input": "audio|embedding" } ],
  "triggers": [ { "id": "...", "on": "<member_id>", "action": "speech_capture", "labels": [...], "min_confidence": 0.5 } ]
}
```

- **`classifiers`**: named backend instances (`yamnet`, `birdnet`, `drone_head`, `stt`).
- **`contexts`**: which classifiers run, always, for a given audio context
  (`detection_trigger`, `localized_render`, `omni_continuous`). Every member in
  `run` executes on that context's audio — this is how BirdNET runs "always on",
  not chained behind YAMNet.

The shipped default leaves `detection_trigger` empty: it is the inexpensive
RMS/cooldown admission gate, not a model-inference context. YAMNet and BirdNET
run on `localized_render` (the beamformed track render); BirdNET also runs on
`omni_continuous`. Operators can add YAMNet to `omni_continuous` in their
routing JSON when that trade-off is desired.

`omni_continuous` uses a normalized sum of each node's synchronized microphone
windows and RMS-gates that mix before inference. The default minimum RMS is
`0.001` (`MINIMAPPR_OMNI_SCAN_MIN_RMS`). A normalized sum preserves the gate's
amplitude scale while retaining the SNR benefit for coherent sources.
- **`chains`**: a stage attached after a parent member. `input: "embedding"`
  feeds the parent's per-frame YAMNet embeddings directly (no re-inference) —
  this is how the drone head rides YAMNet for free in every context where
  YAMNet is a `run` member.
- **`triggers`**: side-effect hooks evaluated by `FusionNode`/the omni scanner,
  not inside classifiers. Currently only `speech_capture`, fired when a member's
  score map contains the given label above `min_confidence` (checked against
  the full scores map, not just the winning label, so a quieter "Speech" score
  under a louder top-1 still triggers capture).

Missing an **optional** backend (birdnet, moonshine) drops that member with one
INFO log. Missing a **required** backend (yamnet, or a drone-head model file on
a non-trimmed install) raises at startup with a remediation message.

## Kill switches

`Settings` fields strip members/chains/triggers from the loaded routing without
editing the JSON: `birdnet_enabled`, `drone_head_enabled`, `stt_enabled`,
`omni_scan_enabled`.

## Power-user API

`GET /api/v1/classifier-routing` returns the canonical routing document along
with its configured path and source (`file` or `default`).

`PUT /api/v1/classifier-routing` accepts `{"routing": { ... }}` containing the
complete document. The server validates it with the same schema used at startup
and atomically replaces the configured file. Its successful response sets
`restart_required: true`: restart Fusion and the Rust classifier helper before
expecting the changed graph to run. Routing is deliberately not hot-reloaded so
model lifecycle work cannot interrupt the audio pipeline.

This API changes the routing document only. The PATCHable kill switches above
remain a separate operational override and may remove configured members,
chains, or triggers from the effective runtime graph.

## Migration from `classifier_backend` / `model_chain.json`

- `classifier_backend`, `model_chain_config_path`, and
  `resolved_classifier_backend()` are gone. Setting `MINIMAPPR_CLASSIFIER` in
  the environment now raises at `Settings.from_env()` with a migration message
  pointing here.
- If `data/model_chain.json` still exists on disk, one ERROR is logged at
  startup pointing at `classifier_routing.json` — it is otherwise ignored.
- Detection provenance (`classifier_backend_name` on `DetectionEvent` /
  `upsert_label`) is now the winning composite member id (`yamnet`, `birdnet`,
  `drone_head`) rather than a single global backend name.

## Training the drone head

`scripts/train_drone_head.py --dataset-dir drone_dataset --out-dir data/models`
trains on `drone_dataset/{drone,no_drone}/*.tfdata` (YAMNet embedding frames)
plus any promoted `data/training/{id}.npy` examples, and writes an int8 QDQ
ONNX model (`drone_head.onnx`) + `drone_head.metadata.json` (labels, alert/detect
thresholds, metrics). Only the int8 artifacts are committed; float32/tflite/
eval-report intermediates are gitignored.

## Speech capture / STT retention

Moonshine (ONNX) transcribes utterances captured by `SpeechCaptureManager`
whenever a routing `speech_capture` trigger fires. Each utterance is written to
`data/speech/{id}.wav` plus a `transcripts` DB row (pre-roll + continuation,
capped by `stt_max_utterance_seconds`). Rows and audio older than
`transcript_retention_seconds` (default 1 week) are deleted by the cleanup
service's housekeeping cycle — this is a flat retention policy, not tiered like
detection/track data, since transcripts are privacy-sensitive by nature.

`GET /api/v1/transcripts?since_ns&limit`, `GET /api/v1/transcripts/{id}/audio`,
and a `{"type":"transcript",...}` WS push cover read access. A `help_me_alert`
rule (`scope: "transcript"`, `transcript_contains: ["help me"]`) fires a
critical alert; the wake-word→LLM path is a future consumer of the same
`SpeechCaptureManager.add_consumer()` seam.
