# W2-core-codec-proto: notes

Branch: `fm/core-codec-proto` (off `field-mode` at `bca0407`).

## What landed (all in `minimappr-core`, new modules only)

| Module | Contents |
|---|---|
| `clip_codec::decimation_filter_design` | Kaiser-windowed-sinc design, response measurement, rate constants |
| `clip_codec::linear_phase_decimator` | `FieldClipDecimator`: 48 to 24 kHz from the rolling buffer, group delay removed, bird rumble cut |
| `clip_codec::flac_clip_codec` | PCM16 to FLAC (`flacenc`) and back (`claxon`); feature `flac-codec` |
| `clip_codec::clip_container` | `MMCL` binary clip header with MMB3-style framing |
| `clip_codec::field_clip` | glue: decimated window to MMCL+FLAC bytes and back, with header/audio consistency checks |
| `stage1::*` | band analyzer, u8 band history, gated noise floor, `Stage1Detector` (contrast/flux trigger, onsets), noise fill |
| `protocol::*` | `FieldMessage` enum (24 kinds, `GroupStatus` reserved at id 24), postcard framing, wire-code enums |
| `session_crypto::*` | XChaCha20-Poly1305 envelopes, replay window, SPAKE2 join-code exchange; feature `session-crypto` |

`lib.rs` only gained `pub mod` lines. The deps sit in a labelled block in `minimappr-core/Cargo.toml`.

## Dependency and feature decision

Every dependency is pure Rust with no C toolchain, so the NDK is not needed:

- `postcard` 1.1 (`default-features = false, features = ["alloc"]`), always on. It is tiny.
- `flacenc` 0.5.1 (`default-features = false`: no `par`/crossbeam threads, no log/serde) and `claxon` 0.4.3 (no deps) sit behind the **`flac-codec`** feature.
- `chacha20poly1305` 0.10 (`alloc` only, so **no `getrandom`**), `spake2` 0.4 (`default-features = false`, so no `getrandom`), `hkdf`, `hmac`, `sha2` and `rand_core` 0.6 sit behind the **`session-crypto`** feature.

Both features are **default-on**, so `cargo test --workspace` and the cross-target builds cover them. A DSP-only consumer can opt out with `default-features = false`. The sidecar inherits the defaults and builds about 15 more small crates; nothing in it uses them. I kept everything in `minimappr-core` rather than a new crate because the deps are light and the protocol types (`NodeId`, `SessionId`) are shared by sealing.

`cargo tree --target wasm32-unknown-unknown` shows no `getrandom`. Randomness is always injected: the crypto functions take a caller's `rand_core` CSPRNG, and noise fill takes a seed. That keeps the crate sans-IO and stops wasm32 from needing JS glue.

## Decimator (FIR) specs achieved

- Anti-alias lowpass: **147 taps**, odd and exactly symmetric, so the group delay is **73 input samples (integer)**. Measured on a 4000-point grid per band: **passband ripple 0.00052 dB p-p (0 to 10.5 kHz), stopband ≥ 90.10 dB (12.5 to 24 kHz)**, DC gain 1. Spec: 90 dB.
- **The group delay is removed by centring the convolution** on device sample `window_first + 2n`, not by shifting afterwards. Tests: an impulse at an even device index peaks at exactly output `k` with a symmetric response. An odd-index impulse gives two equal samples `k` and `k+1`. A 1 kHz sine stays within 4 LSB of the analytic sine evaluated at each output's device index.
- A full-scale-ish 13 kHz chirp tone aliases at ≤ 2 LSB.
- Filter ids (wire-stable):
  - `1 BirdBandWithRumbleCut`: the lowpass plus a linear-phase rumble cut, `x - (boxcar481 * boxcar481) x` at 24 kHz. Its integer delay of 480 output samples is also removed. Response: about -4.4 dB at 25 Hz, -17 dB at 10 Hz, 0.946 at 40 Hz, so grouse drumming survives. It removes DC.
  - `2 FlatToDirectCurrentForDroneAndGunshot`: the lowpass only. DC and a 40 Hz tone pass within 0.2%.
- The window is cut from the rolling buffer with real neighbouring audio as context: `required_context_input_samples_each_side()` = 73 + 960 samples for bird mode. Context outside the buffer is zero-padded, and that is flagged in the window and in the clip header.
- Output is PCM16 and saturation is counted (`clipped_output_sample_count`), never wrapped.

## FLAC

- The round trip is bit-exact on noise at lengths 16, 17, 4095, 4097, 4111, 8193 and 36 001, on silence, on full-scale alternating ±32767/-32768 and extremes, and on stereo.
- **Bug found and fixed:** flacenc writes a final block shorter than 16 samples (for example length 4097 at block size 4096), and claxon then rejects the stream ("block size must be at least 16"). The encoder now picks the largest block size ≤ 4096 whose final block is 0 or ≥ 16 samples. A unit test sweeps every length from 4097 to 2 000 000. Clips under 16 output samples are stored as raw PCM16 (codec id 0) in the container, which is exact and only applies to degenerate windows.
- **Compression on the 9 committed fixtures** (bird filter, whole files, 24 kHz): **1.75× vs 24 kHz PCM16 overall** (per file 1.58× ambience to 2.17× robin), which is **3.49× vs raw 48 kHz PCM16**. That is about **33 to 46 KB per 1.5 s clip**, which matches the plan's 30 to 50 KB estimate.

## Clip container (`MMCL`, version 1)

It follows the MMB3 conventions from `audio_payload.rs`: 4-byte magic and version byte, little-endian fixed fields, then a `u16` section-flag word whose set bits each carry a `u16`-length-delimited section. Unknown sections are skipped, and known sections must parse exactly.

Fixed fields: codec id, decimation filter id, channel count, output rate, device-clock rate, decimation factor, **`u64` first-sample index on the device clock**, and output samples per channel. Section 0x0001 holds PCM16 integrity (clipped count and zero-padded flag). Then come a `u32` length and the encoded audio.

The magic is deliberately not `MMB3`, so `decode_audio_payload_segments` never mistakes a clip for an ingest frame. Firmware and sidecar are untouched.

## Stage-1

- **Bands:** 40 bands at 0.5 to 12 kHz in birds mode (2048-point Hann window, 20 ms hop). Drone mode uses 32 bands at 60 Hz to 6 kHz with an 8192-point window. Bands are mel-like: log-spaced but never narrower than 4 FFT bins (8 in drone mode). One- or two-bin bands are chi-square variables that flicker by several dB in stationary noise. They made drone mode false-trigger in the first run, which is measured, not a guess.
- **History:** a u8 ring holding 30 s × 50 frames/s × 40 bands = 60 KB, quantized in 0.5 dB steps (dBFS).
- **Noise floor:** a gated EMA per band, plus a broadband mean-square floor for noise fill. It tracks symmetrically within ±3 dB, rises very slowly above that so a call cannot lift its own floor, and falls faster below it.
- **Trigger:** the contrast is the excess over the floor of the strongest band in the trigger band (birds 1.5 to 10 kHz, drone 80 Hz to 4 kHz), on 3-frame-smoothed energies (8 frames in drone mode). It opens at ≥ 7 dB for 2 frames (5 in drone mode) and closes 2 dB lower after 10 frames. A sharp onset (flux ≥ 12 dB) also opens it. **There is no absolute level gate.** Tests run the same signals at 0 dB and -40 dB overall gain with identical outcomes, and white noise at -30, -50 and -90 dBFS never triggers.
- **Onsets:** positive flux is computed on 2-frame-smoothed excess, compared with the excess one smoothing span earlier (non-overlapping), minus a 3 dB noise allowance. Peaks ≥ 6 dB are picked with one frame of look-ahead and stamped at the start of their newest hop (±20 ms). They are kept in a bounded list, with `onsets_in_device_range()` for clip payloads.
- **Sensitivity (measured):** a 4 kHz call whose RMS is **10 dB below the broadband noise RMS** (about +8.6 dB in its own band) triggers. The fixture whistle (`wtsp`) mixed **20 dB below** the noise RMS triggers inside the song and never before it. Onsets come reliably at about +13 dB in-band. Tones at 150 to 600 Hz trigger drone mode but not bird mode.
- **False-trigger rate:** in 8 seeds × 120 s of white noise, birds mode opened twice in total (about 1 per 8 minutes). Stage-1 false positives only wake the classifier, so this is cheap; tuning belongs to Phase 5.
- **Counters** for the miss-rate audit: frames evaluated, rejected, triggered, triggers, onsets, onsets dropped and stream discontinuities. `last_frame_diagnostics()` exposes contrast and flux for field tuning.
- **Noise fill:** `blank_span_with_noise_fill` cross-fades over a 2.5 ms raised cosine into uniform noise at the running floor RMS, with a 1-LSB minimum so digital silence still gets non-zero variance (the BirdNET V3.0 NaN guard). It never writes zeros. The noise source is a seeded xorshift.

## Message protocol

- A frame is `postcard(WireHeader{major, minor, kind})` followed by `postcard(body)`. Kind ids 1 to 24 are listed in `protocol::field_message`. They cover join and auth (with node descriptor), the join-code exchange, chirp schedule and arrival reports, island membership, detection events, readiness query and answer, DOA bearings, clip request, payload and chunk, LOCO request and rescores, tap-to-hear request and payload, layout model, clock model, bird-entity updates, queued-detection upload (keyed by `(node_id, local_seq)`), heartbeat (thermal headroom and placement), people-nearby summary, and **`GroupStatus` (24, reserved)**.
- **Forward compatibility:**
  - Another `major` is an error.
  - An unknown `kind` decodes to `DecodedFrame::UnknownKind{body}`, so hosts can log or relay it.
  - Bodies grow only by appending fields, and decoders ignore trailing bytes (`take_from_bytes`).
  - Nested enums travel as `u8` codes with `Unrecognized(u8)`, because postcard is not self-describing and an unknown serde variant would kill the whole message.
  - Kind ids are never reused.
- Sample indices in every message are on the **named node's own** device clock; the host maps them with its clock model. A readiness query encodes to ≤ 24 B (the plan budgets about 20 B).

## Sealing and PAKE: decision note

**Threat model.** The join code is spoken aloud among birders, so it has low entropy (6 digits, about 20 bits) and anyone in earshot may know it. The Cloudflare relay is untrusted: it sees and can drop, reorder, replay, re-address or inject every message, and it can impersonate host or joiner during a join. Passive network attackers are covered by the same measures.

**Sealing.** Messages use XChaCha20-Poly1305, which has a 192-bit nonce, so a random nonce per message needs no counter coordination across host handover. The key is `HKDF-SHA256(salt = session_id, ikm = QR session key)`. The cleartext routing prefix (version, sender, recipient, sender sequence) is AEAD associated data: the relay can route on it but cannot re-address or renumber. Replays are dropped by a per-sender 64-message sliding window (the IPsec/DTLS pattern), which is updated only after authentication.

**PAKE choice: SPAKE2** (RustCrypto `spake2` 0.4, Ed25519 group). It is a balanced PAKE: both sides know the same low-entropy code, which matches "host shows the code, joiner types it". The relay running the exchange gets **one online guess per attempt** and no offline dictionary attack. Pure Rust; wasm32, iOS and Android build.

Alternatives considered:
- **CPace:** cleaner CFRG pick, but no maintained, audited pure-Rust crate.
- **OPAQUE:** an augmented PAKE for stored server passwords; the wrong shape here and much heavier.
- **Plain HKDF of the code:** offline brute force of 20 bits takes milliseconds. Rejected.

**Hardening on top of SPAKE2:**
1. The joiner sends its key-confirmation tag first. Flight 2 (SPAKE2 message B) carries no confirmation, so a fake joiner learns nothing until the host has seen, and counted, a wrong tag. After 5 wrong tags (`DEFAULT_MAXIMUM_FAILED_CONFIRMATIONS`) the code is retired, which bounds an attacker's success to about 5 × 10⁻⁶ per code.
2. Success then comes from AEAD-opening the sealed session key (flight 4), which also authenticates the host.
3. The identities embed the session id, so transcripts do not transfer between sessions.
4. **Shell and relay contract (W3-relay):** the Worker needs some lookup handle to find the session, but **that handle must not be the PAKE password**. Split the join code into a relay-visible lookup part and a secret part that is never sent to the Worker (for example `K7-731904`: `K7` routes, `731904` is the SPAKE2 password). If the Worker saw the whole code, it could run the PAKE itself and win every time.

Caveat: `spake2` 0.4 is the python-spake2/magic-wormhole-compatible variant, not byte-exact RFC 9382. That doesn't matter while both ends run this crate.

## Cross-target builds

| Target | Result |
|---|---|
| host (aarch64-apple-darwin), default features and `--no-default-features` | builds |
| `wasm32-unknown-unknown` | builds (no getrandom in tree) |
| `aarch64-linux-android` | builds (rlib; no NDK needed) |
| `aarch64-apple-ios` | `rustup target add` worked; builds |

## Tests

New in `minimappr-core`:

- `tests/clip_decimation_filter.rs`: 9 tests
- `tests/clip_flac_and_container.rs`: 8 tests (prints the per-fixture ratios with `--nocapture`)
- `tests/stage1_trigger.rs`: 9 tests
- `tests/protocol_messages.rs`: 7 tests
- `tests/session_crypto.rs`: 9 tests
- 1 lib unit test (FLAC block-size sweep)

`tests/common/mod.rs` holds the fixture loader and seeded Gaussian noise.

## Promotion review fixes (review/core-codec-proto)

- `decode_flac_to_pcm16` now takes a mandatory `maximum_interleaved_samples` bound. Before, a few KB of FLAC CONSTANT subframes from a peer could decode to gigabytes. `decode_field_clip` bounds it by the header's sample count and rejects headers over the 30 s rolling buffer (`FIELD_CLIP_MAXIMUM_SAMPLES_PER_CHANNEL`), with more than 8 channels, a zero decimation factor, or a device-clock index that overflows.
- FLAC metadata other than STREAMINFO and PADDING is rejected before claxon sees it, because claxon pre-allocates from VORBIS_COMMENT and APPLICATION length fields (about 80 MB from a few dozen bytes).
- A raw PCM16 clip with an odd byte length is rejected instead of being truncated silently.
- New `MessageSealer::open_addressed_to_local_node` rejects authentic envelopes that were addressed to another node or that reflect the receiver's own messages. The AAD stops the relay from editing routing but not from mis-delivering.
- The band-history ring slot is reduced in u64 (a wasm32 truncation at 2^32 frames).
- Tests cover a FLAC bomb, hostile metadata, hostile container headers, every truncation of a clip, every truncation and corruption of every message kind, huge postcard length prefixes, relay mis-delivery and reflection, and a reflected SPAKE2 flight.

## Open issues / follow-ups

- **Sender sequence across restarts (W3):** the replay window keys on `(sender, sequence)` only. A sender that restarts with sequence 0 is dropped until it passes its old high-water mark. A receiver that restarts accepts replays of older envelopes. The shell must persist the sequence per session (or the envelope needs a sender epoch).
- **Group key:** every member holds the same message key, so a member (not the relay) can forge any sender id. This is acceptable under the relay-only threat model, but per-sender keys or signatures are needed before members are untrusted.
- **Backward compatibility of appended fields:** a newer decoder that reads an older peer's shorter body fails (postcard has no defaults on EOF). Handle this when the first field is appended.
- **Host join DoS (W3):** cap the pending-attempt map. Anyone who can reach the host can spend the code's 5-guess budget with garbage confirmations. That is a denial of service, not a key compromise.

- The Stage-1 thresholds are set from synthetic white noise and one fixture. Real outdoor noise (wind gusts, insects) needs the Phase 5 sampled-miss audit. Pink and wind noise were not tested.
- `stage1_trigger` takes about 16 s in debug builds (drone mode runs 8192-point FFTs at 50 Hz). A `[profile.test.package.minimappr-core] opt-level` override in the root `Cargo.toml` would fix it. I did not touch the root manifest because parallel streams edit it.
- `ClipPayloadChunk` is defined, but chunking and reassembly logic belongs to W3-relay/transport.
- Join-code formatting (lookup/secret split, alphabet) is W3-relay's job; the core takes the secret bytes.
- Decoders accept but do not interpret newer minor versions; a per-kind "minimum minor" table can come when the first appended field lands.
