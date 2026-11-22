# Developer Recipes

Short, copy-pasteable command sequences for recurring development tasks. Run everything from the repo
root with the dev virtualenv active (`source .venv/bin/activate`, see the README "Development" section),
so the relative `data/` paths resolve.

## Rust vs Python ingest/DSP side-by-side

Two fully independent servers, each with its own UI port, ingest port, database, and spool directory, so
the same node traffic can be compared across the Rust and Python ingest backends.

Create the isolated data trees first:

```bash
mkdir -p data/bench_rust data/bench_python
```

Terminal 1, Rust-backed instance (UI on 8080, ingest on 8081). Needs the sidecar binary
(`scripts/build_rust.sh --all`):

```bash
MINIMAPPR_PORT=8080 \
MINIMAPPR_INGEST_PORT=8081 \
MINIMAPPR_INGEST_BACKEND=rust \
MINIMAPPR_DB_PATH=data/bench_rust/minimappr.db \
MINIMAPPR_INGEST_SPOOL_DIR=data/bench_rust/spool \
MINIMAPPR_INGEST_SIDECAR_BINARY_PATH=dist/minimappr-ingest-sidecar \
python -m minimappr serve
```

Terminal 2, Python-backed instance (UI on 8090, ingest on 8091). Because `MINIMAPPR_INGEST_PORT` differs
from `MINIMAPPR_PORT`, this one terminal supervises a separate Python ingest process plus the API:

```bash
MINIMAPPR_PORT=8090 \
MINIMAPPR_INGEST_PORT=8091 \
MINIMAPPR_INGEST_BACKEND=python \
MINIMAPPR_DB_PATH=data/bench_python/minimappr.db \
MINIMAPPR_INGEST_SPOOL_DIR=data/bench_python/spool \
python -m minimappr serve
```

Only the database and spool are isolated. Snippets, speech audio, and training data still default to the
shared `data/` paths, so set `MINIMAPPR_SNIPPET_DIR`, `MINIMAPPR_SPEECH_AUDIO_DIR`, and
`MINIMAPPR_TRAINING_DATASET_DIR` per instance if those outputs need to stay separate.

## Drone head model training

Needs the `train` extra (`pip install -e ".[train]"`). First fetch a deterministic per-class subsample of
the Hugging Face drone dataset as WAVs under `drone_dataset/wav/<label>/`. The label map converts the
dataset's numeric labels; check the actual label values with `--dry-run` first:

```bash
python scripts/fetch_drone_audio.py \
  --repo geronimobasso/drone-audio-detection-samples \
  --out-dir drone_dataset/wav --max-per-class 40000 --seed 1234 \
  --label-map "1=drone,0=ambient"
```

`--seed` is accepted but currently unused, because ranking is hash-deterministic.

Then train the YAMNet-embedding drone head. It also pulls server-promoted examples from `data/training`
and writes the int8 ONNX model and its metadata to `data/models/`:

```bash
python scripts/train_drone_head.py \
  --wav-dir drone_dataset/wav --promoted-dir data/training \
  --out-dir data/models --cache-dir drone_dataset/.embed_cache \
  --labels ambient,drone,coyote --epochs 40 \
  --aug-copies 1 --aug-copies-per-label coyote=8 --synth-ambient-count 1000
```

Put coyote clips and false positives in `drone_dataset/wav/coyote/` and `drone_dataset/wav/ambient/`
respectively. Run `python scripts/train_drone_head.py --help` for the quantization-gate and balancing
options.
