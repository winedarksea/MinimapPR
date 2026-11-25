"""Faint admission funnel through normal ingest (plan item E7).

At the app's default trigger settings (a fixed operating point whose false
trigger rate is measured on noise-only input), each SNR step reports trigger
recall, a usable localization and the stored detection modality, so the
losing stage is visible (``tests/faint_admission_funnel.py``). The table is
printed and written as JSON (``MINIMAPPR_FAINT_FUNNEL_REPORT_PATH`` to keep it).

Quiet-site fixture: a faint call at a site whose floor is MEMS self-noise
stands well above the noise-floor-relative trigger's bar but below both the
default absolute trigger and the solver's old absolute channel floor. Before
R4 (relative trigger OR'd in, relative solver floor) it was never triggered
and was stored omni; it must now be relative-triggered and localized.

Default suite: noise-only at the quiet site, the quiet-site fixture, and two
normal-site steps. Full sweep: ``MINIMAPPR_ENABLE_FAINT_FUNNEL_FULL=1``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from minimappr.config import Settings
from minimappr.core.localization_overhaul_config import (
    DEFAULT_LOCALIZATION_SOLVER_FLOOR_RELATIVE_DB,
    DEFAULT_NOISE_FLOOR_RELATIVE_TRIGGER_SNR_DB,
)
from tests.faint_admission_funnel import (
    APP_DEFAULT_TRIGGER_COOLDOWN_SECONDS,
    APP_DEFAULT_TRIGGER_RMS,
    ATTEMPT_KIND_NOISE_FLOOR_RELATIVE,
    NODE_ID,
    NORMAL_SITE,
    OLD_ABSOLUTE_SOLVER_FLOOR_RMS,
    QUIET_SITE,
    FunnelRow,
    format_funnel_table,
    funnel_captures,
    funnel_scene,
    loudest_event_window_rms,
    run_funnel_step,
    write_funnel_report,
)

FULL_SWEEP_ENV = "MINIMAPPR_ENABLE_FAINT_FUNNEL_FULL"
REPORT_PATH_ENV = "MINIMAPPR_FAINT_FUNNEL_REPORT_PATH"
# Any false trigger in the 12 s noise-only capture is 5/min, over this budget.
FALSE_TRIGGER_BUDGET_PER_MINUTE = 1.0
# In-band SNR of the quiet-site call: ~10 dB over the floor broadband, ~15 %
# under the old 1e-4 RMS solver floor.
QUIET_SITE_FIXTURE_IN_BAND_SNR_DB = 19.0
# The server sees PCM16; quantisation lifts a -92 dBFS floor by ~0.5 dB.
PCM16_FULL_SCALE = 32768.0
# Steady white noise leaves the relative trigger's fluctuation term near 0 dB;
# this margin keeps the fixture clear of the bar anyway.
RELATIVE_TRIGGER_MARGIN_DB = 1.5
# Normal site: 24 dB sits under the default absolute gate but ~15 dB over the
# floor (relative trigger territory); 34 dB clears both.
NORMAL_SITE_RELATIVE_ONLY_SNR_DB = 24.0
NORMAL_SITE_ABOVE_ABSOLUTE_GATE_SNR_DB = 34.0
FULL_SWEEP_SNR_STEPS_DB = (0.0, 6.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 38.0)


class FaintEventLost(AssertionError):
    """Raised only by funnel verdicts, so a funnel loss reads apart from a harness failure."""


def _report(rows: list[FunnelRow], tmp_path: Path, name: str) -> None:
    print(f"\n[faint-funnel] {name}\n{format_funnel_table(rows)}")
    write_funnel_report(rows, tmp_path / f"{name}.json")
    report_dir = os.environ.get(REPORT_PATH_ENV)
    if report_dir:
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        write_funnel_report(rows, Path(report_dir) / f"{name}.json")


def _assert_fully_admitted(row: FunnelRow) -> None:
    if row.losing_stage is not None:
        raise FaintEventLost(f"lost at {row.losing_stage}:\n{format_funnel_table([row])}")


def test_funnel_uses_the_apps_default_trigger(monkeypatch):
    monkeypatch.delenv("MINIMAPPR_TRIGGER_RMS", raising=False)
    monkeypatch.delenv("MINIMAPPR_TRIGGER_COOLDOWN_SECONDS", raising=False)
    settings = Settings.from_env()
    assert settings.trigger_rms == APP_DEFAULT_TRIGGER_RMS
    assert settings.trigger_cooldown_seconds == APP_DEFAULT_TRIGGER_COOLDOWN_SECONDS


def test_quiet_site_fixture_sits_between_the_relative_trigger_and_the_old_absolute_floor():
    """Harness proof for the live fixture below, from the synthesized capture alone."""
    scene = funnel_scene()
    capture = funnel_captures(scene, QUIET_SITE, QUIET_SITE_FIXTURE_IN_BAND_SNR_DB)[NODE_ID]
    received = np.round(capture.astype(np.float64) * PCM16_FULL_SCALE) / PCM16_FULL_SCALE
    pre_roll_floor_rms = float(np.sqrt(np.mean(received[:, : received.shape[1] // 4] ** 2)))
    loudest_rms = loudest_event_window_rms(scene, received)
    over_floor_db = 20.0 * np.log10(loudest_rms / pre_roll_floor_rms)
    assert pre_roll_floor_rms == pytest.approx(QUIET_SITE.sensor_noise_rms, rel=0.1)
    assert loudest_rms < OLD_ABSOLUTE_SOLVER_FLOOR_RMS < APP_DEFAULT_TRIGGER_RMS
    assert over_floor_db >= DEFAULT_NOISE_FLOOR_RELATIVE_TRIGGER_SNR_DB + RELATIVE_TRIGGER_MARGIN_DB
    # A relative solver floor admits anything the relative trigger admits.
    assert DEFAULT_LOCALIZATION_SOLVER_FLOOR_RELATIVE_DB < DEFAULT_NOISE_FLOOR_RELATIVE_TRIGGER_SNR_DB


def test_noise_only_false_trigger_rate_at_the_quiet_site(monkeypatch, tmp_path: Path):
    row = run_funnel_step(monkeypatch, tmp_path, QUIET_SITE, None)
    _report([row], tmp_path, "noise_only_quiet")
    assert row.false_triggers_per_minute <= FALSE_TRIGGER_BUDGET_PER_MINUTE, format_funnel_table([row])


def test_quiet_site_faint_call_is_relative_triggered_and_localized(monkeypatch, tmp_path: Path):
    row = run_funnel_step(monkeypatch, tmp_path, QUIET_SITE, QUIET_SITE_FIXTURE_IN_BAND_SNR_DB)
    _report([row], tmp_path, "quiet_site_fixture")
    if ATTEMPT_KIND_NOISE_FLOOR_RELATIVE not in row.trigger_kinds:
        raise FaintEventLost(f"not admitted by the relative trigger:\n{format_funnel_table([row])}")
    _assert_fully_admitted(row)


@pytest.mark.parametrize(
    "in_band_snr_db",
    [
        pytest.param(NORMAL_SITE_RELATIVE_ONLY_SNR_DB, id="relative_only"),
        pytest.param(NORMAL_SITE_ABOVE_ABSOLUTE_GATE_SNR_DB, id="above_absolute_gate"),
    ],
)
def test_normal_site_funnel_step(monkeypatch, tmp_path: Path, in_band_snr_db: float):
    row = run_funnel_step(monkeypatch, tmp_path, NORMAL_SITE, in_band_snr_db)
    _report([row], tmp_path, f"normal_site_{in_band_snr_db:g}db")
    _assert_fully_admitted(row)


@pytest.mark.skipif(os.environ.get(FULL_SWEEP_ENV) != "1", reason=f"full funnel sweep is opt-in ({FULL_SWEEP_ENV}=1)")
def test_full_funnel_sweep(monkeypatch, tmp_path: Path):
    """Report-only per step; gated on the false-trigger budget and on the top
    normal-site step (above every gate) being fully admitted."""
    rows: list[FunnelRow] = []
    for site in (NORMAL_SITE, QUIET_SITE):
        for in_band_snr_db in (None, *FULL_SWEEP_SNR_STEPS_DB):
            step_dir = tmp_path / f"{site.name}_{in_band_snr_db}"
            step_dir.mkdir()
            with monkeypatch.context() as step_monkeypatch:
                rows.append(run_funnel_step(step_monkeypatch, step_dir, site, in_band_snr_db))
    _report(rows, tmp_path, "full_sweep")
    noise_rows = [row for row in rows if row.false_triggers_per_minute is not None]
    assert all(row.false_triggers_per_minute <= FALSE_TRIGGER_BUDGET_PER_MINUTE for row in noise_rows), format_funnel_table(noise_rows)
    top_normal = next(row for row in rows if row.site == NORMAL_SITE.name and row.in_band_snr_db == FULL_SWEEP_SNR_STEPS_DB[-1])
    _assert_fully_admitted(top_normal)
