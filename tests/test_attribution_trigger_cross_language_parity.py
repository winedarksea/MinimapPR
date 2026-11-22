"""Parity: Rust S5 port (relative trigger, attribution cascade, LOCO decision) vs Python goldens.

Goldens (``tests/fixtures/attribution_trigger_parity_goldens.npz``) come from
``python -m tests.attribution_trigger_parity_cases``; Rust runs through the
sidecar's hidden ``attribution-oracle``. The host-owned masking and classifier
rerun are scripted identically on both sides.

Tolerances:

- **Decisions exact**: trigger booleans per frame, tracked-node count and LRU
  eviction, attribution tier, gradient and LOCO reasons, interferer counts,
  labels, collapsed labels, the shared per-window rerun budget. Dominance is
  compared as the reference annotates it (rounded to 0.01 dB), exactly.
- **Floor RMS and confidences 1e-9 relative**: the floor is a one-pole
  recursion over ``exp``/``log10``; libm differences stay ~1e-15 per step and
  the recursion is contracting, so 600 frames remain far inside 1e-9.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from tests.attribution_trigger_parity_cases import GOLDEN_PATH, all_cases
from tests.parity_json_comparison import assert_json_matches
from tests.rust_sidecar_binary import build_rust_sidecar_binary, cargo_is_available

RELATIVE_TOLERANCE = 1.0e-9


def _load_goldens() -> tuple[list[dict], list[dict]]:
    with np.load(GOLDEN_PATH) as archive:
        cases = [json.loads(str(raw)) for raw in archive["case_inputs_json"]]
        expected = [json.loads(str(raw)) for raw in archive["expected_outputs_json"]]
    return cases, expected


@pytest.mark.skipif(not cargo_is_available(), reason="cargo is required to build the Rust sidecar oracle")
def test_rust_attribution_and_trigger_match_python_goldens() -> None:
    cases, expected = _load_goldens()
    completed = subprocess.run(
        [str(build_rust_sidecar_binary()), "attribution-oracle"],
        input=json.dumps({"cases": cases}),
        capture_output=True,
        text=True,
        check=True,
    )
    results = json.loads(completed.stdout)["results"]
    assert len(results) == len(cases)
    for index, (case, result, golden) in enumerate(zip(cases, results, expected)):
        assert_json_matches(result, golden, f"case {index} ({case['kind']})", RELATIVE_TOLERANCE)


def test_python_reference_still_matches_goldens() -> None:
    cases, expected = _load_goldens()
    regenerated = all_cases()
    assert [json.dumps(case, sort_keys=True) for case, _ in regenerated] == [
        json.dumps(case, sort_keys=True) for case in cases
    ], "golden inputs drifted: rerun python -m tests.attribution_trigger_parity_cases"
    for index, ((case, fresh), golden) in enumerate(zip(regenerated, expected)):
        assert_json_matches(json.loads(json.dumps(fresh)), golden, f"case {index} ({case['kind']})", RELATIVE_TOLERANCE)
