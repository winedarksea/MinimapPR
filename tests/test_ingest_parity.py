"""Cross-language ingest parity — Python-side coverage of the new surfaces.

This is the Python half of the "Item 6" parity corpus from the realtime
ingest plan. It verifies behavior on the Python side that mirrors the
Rust-side unit tests in `minimappr-ingest-sidecar/src/dsp_worker_tests.rs`:

- The new ``stages`` JSON shape on ``NodeAudioOverride`` round-trips through
  ``build_chain_from_rust_stages`` and produces a working chain.
- The Python preprocessing chain produces the same *qualitative* response
  as the Rust biquad cascade (passband preserved, stopband attenuated).
- The Python ``_IngestConcurrencyLimit`` admits up to its ceiling and then
  sheds with HTTP 503 + ``Retry-After``, matching the sidecar's bounded
  MPSC + 503 behavior at ``main.rs::raw_audio_ingest``.

The fully cross-language *subprocess* parity lane (Python ⇄ real Rust
binary) is deferred to its own follow-up — that requires the
``test_rust_manifest_handoff_e2e.py`` subprocess fixture to be extended
with a preprocessing-config endpoint, which is its own substantial chunk
of plumbing.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException

from minimappr.core.preprocessing import (
    AudioPreprocessingChain,
    BandpassFilterStage,
    GainStage,
    HighpassFilterStage,
    LowpassFilterStage,
    build_chain_from_rust_stages,
)
from minimappr.main import _DEFAULT_INGEST_MAX_CONCURRENT, _IngestConcurrencyLimit
from minimappr.models import NodeAudioOverride


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine_wave(frequency_hz: float, sample_rate_hz: int, samples: int) -> np.ndarray:
    """Mirror the Rust `sine_wave` helper in dsp_worker_tests.rs so the parity
    cases use identical input signals on both sides."""
    indices = np.arange(samples, dtype=np.float64)
    return np.sin(2.0 * np.pi * frequency_hz * indices / float(sample_rate_hz)).astype(np.float32)


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


# ---------------------------------------------------------------------------
# build_chain_from_rust_stages — JSON shape parity with Rust PreprocessStage
# ---------------------------------------------------------------------------


class TestBuildChainFromRustStages:
    """The Rust ``PreprocessStage`` JSON shape must build an equivalent Python chain."""

    def test_empty_chain_means_passthrough(self) -> None:
        chain = build_chain_from_rust_stages([])
        assert isinstance(chain, AudioPreprocessingChain)
        assert chain.stages == []

        sr = 16_000
        signal = _sine_wave(440.0, sr, 512)
        out = chain.process(signal.copy(), sr)
        np.testing.assert_array_equal(out, signal)

    def test_passthrough_variant_skipped(self) -> None:
        chain = build_chain_from_rust_stages([{"type": "passthrough"}])
        assert chain.stages == []

    def test_gain_stage_from_db(self) -> None:
        chain = build_chain_from_rust_stages([{"type": "gain", "db": 6.0}])
        assert len(chain.stages) == 1
        assert isinstance(chain.stages[0], GainStage)
        # +6 dB ≈ ×1.9953
        expected_multiplier = 10.0 ** (6.0 / 20.0)
        assert chain.stages[0].multiplier == pytest.approx(expected_multiplier, abs=1e-6)

    def test_highpass_lowpass_stages(self) -> None:
        chain = build_chain_from_rust_stages([
            {"type": "highpass", "cutoff_hz": 200.0, "order": 4},
            {"type": "lowpass", "cutoff_hz": 4_000.0, "order": 2},
        ])
        assert isinstance(chain.stages[0], HighpassFilterStage)
        assert chain.stages[0].cutoff_hz == 200.0
        assert chain.stages[0].order == 4
        assert isinstance(chain.stages[1], LowpassFilterStage)
        assert chain.stages[1].cutoff_hz == 4_000.0
        assert chain.stages[1].order == 2

    def test_bandpass_stage(self) -> None:
        chain = build_chain_from_rust_stages([
            {"type": "bandpass", "low_hz": 300.0, "high_hz": 3000.0, "order": 4},
        ])
        assert isinstance(chain.stages[0], BandpassFilterStage)
        assert chain.stages[0].low_hz == 300.0
        assert chain.stages[0].high_hz == 3_000.0

    def test_dc_block_maps_to_dc_removal(self) -> None:
        chain = build_chain_from_rust_stages([{"type": "dc_block"}])
        assert len(chain.stages) == 1
        # DCRemovalStage is the closest Python equivalent. (The Rust side uses
        # a one-pole IIR, the Python side a single-mean subtraction — they
        # converge for stationary-DC test inputs.)
        from minimappr.core.preprocessing import DCRemovalStage
        assert isinstance(chain.stages[0], DCRemovalStage)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown PreprocessStage type"):
            build_chain_from_rust_stages([{"type": "no_such_filter"}])

    def test_default_order_when_omitted(self) -> None:
        chain = build_chain_from_rust_stages([{"type": "highpass", "cutoff_hz": 100.0}])
        # Default order matches the Rust default (4).
        assert chain.stages[0].order == 4


# ---------------------------------------------------------------------------
# NodeAudioOverride.stages — Pydantic model round-trip
# ---------------------------------------------------------------------------


class TestNodeAudioOverrideStages:
    """The new ``stages`` field on ``NodeAudioOverride`` carries the same JSON
    shape as the Rust ``NodeAudioConfig.stages`` field."""

    def test_stages_field_default_is_none(self) -> None:
        override = NodeAudioOverride()
        assert override.stages is None

    def test_stages_round_trip(self) -> None:
        json_in = {
            "stages": [
                {"type": "gain", "db": 3.0},
                {"type": "highpass", "cutoff_hz": 80.0, "order": 4},
            ]
        }
        override = NodeAudioOverride.model_validate(json_in)
        assert override.stages == json_in["stages"]
        # And serializes back to the same shape.
        assert override.model_dump()["stages"] == json_in["stages"]

    def test_legacy_fields_coexist_with_stages(self) -> None:
        override = NodeAudioOverride.model_validate(
            {"hp_hz": 100.0, "stages": [{"type": "gain", "db": 6.0}]}
        )
        assert override.hp_hz == 100.0
        assert override.stages == [{"type": "gain", "db": 6.0}]


# ---------------------------------------------------------------------------
# Preprocessing chain — qualitative parity with the Rust biquad cascade.
#
# The Rust unit tests in dsp_worker_tests.rs assert that:
#   - lowpass attenuates 4 kHz tone to <10% RMS at fc=500 Hz, order=4
#   - highpass attenuates 100 Hz tone to <10% RMS at fc=500 Hz, order=4
# We assert the same on the Python side. (Bit-exact agreement is not
# expected — Python uses scipy.signal.sosfilt while Rust uses an in-tree
# biquad; both are causal Butterworth but with subtly different design
# math. The qualitative response is what matters for parity.)
# ---------------------------------------------------------------------------


class TestPreprocessingResponseMatchesRustExpectations:
    SR: int = 16_000
    N: int = 4_096
    SETTLE: int = 1_024

    def test_lowpass_passes_low_attenuates_high(self) -> None:
        chain = build_chain_from_rust_stages([
            {"type": "lowpass", "cutoff_hz": 500.0, "order": 4},
        ])
        # 100 Hz should pass (well below 500 Hz cutoff).
        low_in = _sine_wave(100.0, self.SR, self.N)
        low_out = chain.process(low_in.copy(), self.SR)
        low_atten = _rms(low_out[self.SETTLE:]) / _rms(low_in[self.SETTLE:])

        # Fresh chain so state doesn't carry over.
        chain = build_chain_from_rust_stages([
            {"type": "lowpass", "cutoff_hz": 500.0, "order": 4},
        ])
        hi_in = _sine_wave(4_000.0, self.SR, self.N)
        hi_out = chain.process(hi_in.copy(), self.SR)
        hi_atten = _rms(hi_out[self.SETTLE:]) / _rms(hi_in[self.SETTLE:])

        assert low_atten > 0.9, f"100 Hz tone unexpectedly attenuated: {low_atten:.3f}"
        assert hi_atten < 0.1, f"4 kHz tone not attenuated enough: {hi_atten:.3f}"

    def test_highpass_passes_high_attenuates_low(self) -> None:
        chain = build_chain_from_rust_stages([
            {"type": "highpass", "cutoff_hz": 500.0, "order": 4},
        ])
        low_in = _sine_wave(100.0, self.SR, self.N)
        low_out = chain.process(low_in.copy(), self.SR)
        low_atten = _rms(low_out[self.SETTLE:]) / _rms(low_in[self.SETTLE:])

        chain = build_chain_from_rust_stages([
            {"type": "highpass", "cutoff_hz": 500.0, "order": 4},
        ])
        hi_in = _sine_wave(4_000.0, self.SR, self.N)
        hi_out = chain.process(hi_in.copy(), self.SR)
        hi_atten = _rms(hi_out[self.SETTLE:]) / _rms(hi_in[self.SETTLE:])

        assert low_atten < 0.1, f"100 Hz not attenuated enough by HPF: {low_atten:.3f}"
        assert hi_atten > 0.9, f"4 kHz unexpectedly attenuated by HPF: {hi_atten:.3f}"

    def test_filter_state_persists_across_calls(self) -> None:
        """Split-call equivalence: splitting a buffer in two and applying
        the chain to each half should produce the same trailing output as
        applying it to the whole buffer at once. This is the per-stream-state
        invariant that the Rust side asserts in
        `cascade_preserves_state_across_frame_boundaries`."""
        chain_whole = build_chain_from_rust_stages([
            {"type": "highpass", "cutoff_hz": 200.0, "order": 4},
        ])
        chain_split = build_chain_from_rust_stages([
            {"type": "highpass", "cutoff_hz": 200.0, "order": 4},
        ])

        full = _sine_wave(50.0, self.SR, 1_024)
        whole_out = chain_whole.process(full.copy(), self.SR)
        first_out = chain_split.process(full[:512].copy(), self.SR)
        second_out = chain_split.process(full[512:].copy(), self.SR)
        stitched = np.concatenate([first_out, second_out])

        # Allow small floating-point tolerance; the state-carry should yield
        # near-identical numerical output.
        np.testing.assert_allclose(stitched, whole_out, atol=1e-5)


# ---------------------------------------------------------------------------
# Concurrency limiter — admits to ceiling, sheds with 503 + Retry-After.
# ---------------------------------------------------------------------------


class TestIngestConcurrencyLimit:
    def test_default_ceiling_constant_is_nonzero(self) -> None:
        assert _DEFAULT_INGEST_MAX_CONCURRENT > 0

    async def _enter(self, limit: _IngestConcurrencyLimit) -> None:
        await limit.__aenter__()

    async def _exit(self, limit: _IngestConcurrencyLimit) -> None:
        await limit.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_admits_below_ceiling(self) -> None:
        limit = _IngestConcurrencyLimit(max_concurrent=3)
        await self._enter(limit)
        await self._enter(limit)
        await self._enter(limit)
        assert limit.active == 3
        assert limit.total_admissions == 3
        assert limit.total_shed == 0

    @pytest.mark.asyncio
    async def test_sheds_at_ceiling_with_retry_after_header(self) -> None:
        limit = _IngestConcurrencyLimit(max_concurrent=2)
        await self._enter(limit)
        await self._enter(limit)
        with pytest.raises(HTTPException) as excinfo:
            await self._enter(limit)
        assert excinfo.value.status_code == 503
        assert excinfo.value.headers is not None
        assert excinfo.value.headers.get("Retry-After") == "1"
        assert limit.total_shed == 1
        assert limit.active == 2  # rejected admission did not bump counter

    @pytest.mark.asyncio
    async def test_release_allows_subsequent_admit(self) -> None:
        limit = _IngestConcurrencyLimit(max_concurrent=1)
        await self._enter(limit)
        await self._exit(limit)
        # Slot freed — next admit succeeds.
        await self._enter(limit)
        assert limit.active == 1

    @pytest.mark.asyncio
    async def test_release_below_zero_is_clamped(self) -> None:
        """A stray release without a matching admit must not push active negative —
        otherwise the counter could become a permanent admission credit."""
        limit = _IngestConcurrencyLimit(max_concurrent=2)
        await self._exit(limit)  # spurious release
        assert limit.active == 0

    @pytest.mark.asyncio
    async def test_exception_during_admit_still_releases(self) -> None:
        """`async with` must call __aexit__ even when the wrapped block raises —
        otherwise an HTTP-500 path could leak a permanent admission."""
        limit = _IngestConcurrencyLimit(max_concurrent=1)
        try:
            async with limit:
                raise RuntimeError("simulated handler crash")
        except RuntimeError:
            pass
        assert limit.active == 0
        # Slot must be reusable.
        async with limit:
            assert limit.active == 1
