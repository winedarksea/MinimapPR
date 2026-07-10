# Beamformed render contract (Rust ↔ Python)

Every classification-bound audio render steered toward a localized source follows
this contract, in both the Rust ingest sidecar (live per-node render) and the
Python fusion node (classification orchestrator, cross-node beams, IAMF objects).
The goal is recall-biased: the classifier must receive the *complete sound
profile* of the source — beamformed where the array physics permit, omni
everywhere else. "A bit too much audio" always beats "too little".

Canonical implementations live in mirrored files; **keep them in sync**:

- Rust: [`src/dsp_math.rs`](src/dsp_math.rs) (shared DSP helpers) and
  [`src/birdnet_render.rs`](src/birdnet_render.rs) (`render_band_split`)
- Python: [`minimappr/core/beamforming.py`](../minimappr/core/beamforming.py)
  (`BandSplitDasRenderer`, `raised_cosine_band_weights`)

## 1. Steering model

Near-field point-source steering everywhere:

```
τ_m = |p_sensor_m − p_steer| / c        (min-subtracted across sensors)
```

(Python `_steering_delays_s`, Rust `steering_delays_s` in `dsp_math.rs`.)

Steer-position rule (which point to steer at), keyed on the localization's
canonical `range_projection_mode` (see `RANGE_PROJECTION_CONTRACT.md`):

- `range_refined` → steer at the solved `position_m` (`steering_model = "near_field"`).
- Range-unobservable modes (`range_asymptotic`, `range_boundary`,
  `range_bearing_projected`) → steer at `direction × projection_range_m`
  (`steering_model = "bearing_projected"`). Near-field steering converges to
  plane-wave at large r, so a single formula covers both cases — this closes the
  historical Rust plane-wave vs. Python point-source parity gap.

## 2. Band split

The physical limit: a discrete array spatially aliases above
`alias_cutoff_hz = c / (2 · max_baseline)`. Beam steering is only valid below
that cutoff; above it, steered summing scrambles the spectrum instead of
focusing it.

- Steered band: `[render_highpass_hz (default 100 Hz), alias_cutoff_hz]`.
- Omni (mean of channels) everywhere else — **including below the highpass**,
  so low rumble is kept for the classifier rather than discarded.
- `alias_cutoff_hz` is computed **from the actual registered sensor geometry at
  render time** (`alias_cutoff_from_positions`), never hardcoded. For the Sirith
  tetra (50 mm edges) this is ≈ 3432 Hz.
- Rust's legacy hardcoded `[1000, 3400]` band becomes an optional clamp:
  effective upper edge = `min(band_max_clamp_hz, alias_cutoff_hz)` when the
  clamp is configured.

## 3. Crossover

Raised-cosine blend, never a hard bin swap:

```
y(f) = w(f) · steered(f) + (1 − w(f)) · omni(f)
```

- Low edge: ramp from 0→1 over `[T_lo − W_lo/2, T_lo + W_lo/2]` with
  `T_lo = render_highpass_hz = 100 Hz` (default width 100 Hz).
- High edge: ramp from 1→0 **centered at** `alias_cutoff_hz`, width
  `T_hi = max(400 Hz, 0.15 · alias_cutoff_hz)`. Centering at the cutoff is the
  recall-biased choice: half the ramp extends above the strict alias limit,
  trading a little aliased energy for continuity of the target's spectrum.
- The legacy Rust hard bin swap (`frequency_blend`) is retired; documented here
  as legacy only.

Reference weight function: `raised_cosine_band_weights(freqs, t_lo, w_lo, t_hi, w_hi)`
returning `w(f) ∈ [0, 1]`, `0.5·(1 − cos(π·x))` over each ramp.

## 4. Reference algorithm (normative)

One rFFT per channel, single inverse FFT:

1. `S_m = rfft(x_m)` for each channel m.
2. `omni(f) = mean_m S_m(f)`.
3. `steered(f) = mean_m S_m(f) · exp(+j 2π f τ_m)` — phase *advance*, matching
   Python `FrequencyDomainDelayAndSumBeamformer`.
4. `y(f) = w(f)·steered(f) + (1−w(f))·omni(f)`; `y = irfft(y(f), n)`.

This is cheaper than the previous Rust live render (time-domain fractional
interpolation + separate FFT/IFFT blend pass).

## 5. Parity tolerance

Rust runs f32, Python f64. Cross-language oracle tests
(`tests/test_beamform_cross_language_parity.py` ↔ `beamform-oracle` CLI mode)
assert:

- Relative RMS error of the output waveform ≤ 1e-3.
- In-band spectral magnitude agreement within 0.5 dB.

## 6. Provenance

Renders under this contract emit:

- `render_kind = "birdnet_band_split_das"` (legacy kinds
  `birdnet_hybrid_spatial_blend` / `birdnet_omni_fallback` remain accepted by
  Python consumers during rollout).
- `effective_spatial_band` = `[low_hz, high_hz]` actually steered after clamping.
- `alias_cutoff_hz` = geometry-derived cutoff used.
- `steering_model` = `"near_field"` | `"bearing_projected"` per rule 1.

## 7. Degradation ladder

For classification of a localized event (documented here; implemented in the
Python fusion node):

1. Multi-node late fusion — per-node band-split beams classified independently,
   label confidences fused by max-with-evidence. No cross-node audio mixing:
   coherent summation is rejected on physics (meter baselines → coherent gain
   only below ~tens of Hz, destroyed by ±1-sample sync error), incoherent
   summation is rejected because it blends noise floors and reverberation.
2. Single-node band-split beam (this contract).
3. Omni reference channel — always computed, always the selection floor: the
   beamformed candidate wins only when its classifier confidence exceeds
   omni's by the configured margin.

## 8. Trajectory rendering headroom (design note)

The renderer primitive is `(windows, steer_position_per_block)`:
`BlockTrajectoryRenderer` (Python, `core/beamforming.py`) drives any duck-typed
beamformer with a per-block steer callback. A continuous per-track stream
(future speech-to-text feed) is the same primitive driven by a `TrackTrajectory`
— no streaming plumbing is built until needed.

Asymmetry note: IAMF *object* rendering uses MVDR with the configured
`mvdr_diagonal_loading` **without** the classifier ×10 recall widening — object
rendering wants spatial selectivity; classification wants recall.
