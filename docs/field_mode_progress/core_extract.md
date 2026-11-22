# W1-core-extract — notes

Branch: `fm/core-extract` (off `field-mode`).

## What changed

- New root `Cargo.toml` workspace: members `minimappr-core`, `minimappr-ingest-sidecar`.
  `Cargo.lock` moved from the sidecar to the root unchanged (only the new crate's entry
  was added; no version drift). `target/` is now the root `target/`.
- `minimappr-frontend` is **excluded**. It is wasm32-only (Leptos/trunk) and has its own
  `[profile.release]` (`opt-level = "s"`, `lto`). In a workspace that profile would have to
  move to the root and would then apply to the native sidecar, and `cargo test --workspace`
  on the host would try to build its web-sys/wasm-bindgen graph natively. Excluded, it keeps
  its own `Cargo.lock` and `target/`, and `trunk build` behaves exactly as before. It can
  still path-depend on `minimappr-core` later (W4-pwa).
- The sidecar defined no `[profile.*]` sections, so the root declares none. A
  `[profile.test] opt-level = 3` for `minimappr-sim` would go in the root when it joins.
- Room is left (comment in root `Cargo.toml`) for `minimappr-sim` and `minimappr-transport`.

## What moved into `minimappr-core`

| Item | From | To |
|---|---|---|
| DSP helpers | `sidecar/src/dsp_math.rs` | `minimappr-core/src/dsp_math.rs` |
| GCC-PHAT | `sidecar/src/gcc_phat.rs` | `minimappr-core/src/gcc_phat.rs` |
| `PairTdoa` | `sidecar/src/dsp_worker.rs` | `minimappr-core/src/gcc_phat.rs` |
| Range-projection constants/prior | `sidecar/src/range_projection.rs` | `minimappr-core/src/range_projection.rs` |
| SRP-PHAT | `sidecar/src/srp_phat.rs` | `minimappr-core/src/srp_phat.rs` |
| `SensorStreamBuffer`, coverage stats | `sidecar/src/dsp.rs` | `minimappr-core/src/sensor_stream_buffer.rs` |
| Store-forward / MMB payload parser | `sidecar/src/audio_payload.rs` | `minimappr-core/src/audio_payload.rs` |
| `butter_lowpass_sos`, `butter_highpass_sos`, `BiquadCoefficients`, `BiquadState` | `sidecar/src/dsp_worker.rs` | `minimappr-core/src/butterworth_biquad.rs` |
| `SIRITH_MIC_POSITIONS_M` | `sidecar/src/dsp_worker.rs` | `minimappr-core/src/sirith_array_geometry.rs` |
| `capture_plane` criterion bench | `sidecar/benches/` | `minimappr-core/benches/` (it only measured moved code) |

Unit tests moved with their modules (67 now run in `minimappr-core`, 116 stay in the sidecar;
183 total, same as before).

The sidecar re-imports the modules at its crate root
(`use minimappr_core::{audio_payload, dsp_math, range_projection, sensor_stream_buffer as dsp, srp_phat}`)
and `dsp_worker.rs` re-exports `PairTdoa`, `SIRITH_MIC_POSITIONS_M` and the biquad items, so
every existing `crate::...` path still resolves. Oracle subcommands in `main.rs` are unchanged.

## Dependency contract

`minimappr-core`: std, serde, rustfft, num-complex, **plus base64 and serde_json** (deviation:
`audio_payload.rs` decodes the JSON store-forward envelope and base64 PCM at runtime; both are
pure Rust and build for wasm32). `tracing` is an optional feature (`tracing`) that only gates
the two `SensorStreamBuffer` re-anchor warnings; the sidecar enables it so its logs are
unchanged. No tokio, axum, rusqlite or rayon.

Builds checked: host, `wasm32-unknown-unknown` (with and without `tracing`),
`aarch64-linux-android` (lib only; no NDK link step needed for an rlib). iOS target not installed.

## Python side

- New `tests/rust_sidecar_binary.py`: builds the sidecar from the repo root with
  `cargo build --message-format=json -p minimappr-ingest-sidecar` and returns the executable
  path Cargo reports, so it works with the workspace `target/`, `CARGO_TARGET_DIR`, or any
  `.cargo/config.toml` override. Six tests that hard-coded
  `minimappr-ingest-sidecar/target/debug/...` now use it.
- `tests/test_range_projection.py` reads `minimappr-core/src/range_projection.rs`.
- `scripts/build_rust.sh` builds from the root and asks `cargo metadata` for the target dir.
- Path mentions in docstrings/contract docs updated to `minimappr-core/src/...`.

## Open items

- `cargo fmt --check` reports pre-existing formatting drift in `gcc_phat.rs` test code
  (left alone so the moved files stay near-verbatim and `git log --follow` tracks them).
- Not done (out of scope): generalizing `srp_phat.rs` beyond tetra geometry (W2-core-srp),
  `no_std`, iOS target build.
