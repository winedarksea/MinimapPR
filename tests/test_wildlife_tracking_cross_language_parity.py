"""Parity: Rust ``wildlife_tracking`` + ``bird_entity`` (S3/S6 port) vs Python goldens.

Goldens (``tests/fixtures/wildlife_tracking_parity_goldens.npz``) come from the
Python reference via ``python -m tests.wildlife_tracking_parity_cases``; the
Rust side runs through the sidecar's hidden ``wildlife-tracking-oracle``.

Tolerances, per component:

- **Decisions are exact**: track/entity ids, association choice, relocation
  and revert events, species-gate and concurrency verdicts, reporting actions
  (insert / enrich / upgrade), entity change kinds and spatial states, labels,
  counts. These are integer or branch outcomes of the same comparisons, so
  any difference is a porting bug.
- **Numbers are 1e-9 relative** (absolute floor 1e-9 × the array's largest
  magnitude): beliefs, filter positions/covariances, bearings, fan widths,
  Mahalanobis scores. Both sides evaluate the same closed-form expressions in
  f64; only LU/eigen solver rounding differs (numpy LAPACK vs nalgebra),
  which stays ~1e-12 on these well-conditioned 2×2/3×3 problems.

A second test re-runs Python against the goldens so a server-side change
shows up here rather than as a mysterious Rust failure.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from tests.parity_json_comparison import assert_json_matches
from tests.rust_sidecar_binary import build_rust_sidecar_binary, cargo_is_available
from tests.wildlife_pipeline_reference import run_pipeline_case
from tests.wildlife_tracking_parity_cases import GOLDEN_PATH, all_cases

RELATIVE_TOLERANCE = 1.0e-9


def _load_goldens() -> tuple[list[dict], list[dict]]:
    with np.load(GOLDEN_PATH) as archive:
        cases = [json.loads(str(raw)) for raw in archive["case_inputs_json"]]
        expected = [json.loads(str(raw)) for raw in archive["expected_outputs_json"]]
    return cases, expected


def _case_label(index: int, case: dict) -> str:
    return f"case {index} ({case['kind']}{':' + case['name'] + '/' + str(case['seed']) if 'name' in case else ''})"


@pytest.mark.skipif(not cargo_is_available(), reason="cargo is required to build the Rust sidecar oracle")
def test_rust_wildlife_tracking_matches_python_goldens() -> None:
    cases, expected = _load_goldens()
    completed = subprocess.run(
        [str(build_rust_sidecar_binary()), "wildlife-tracking-oracle"],
        input=json.dumps({"cases": cases}),
        capture_output=True,
        text=True,
        check=True,
    )
    results = json.loads(completed.stdout)["results"]
    assert len(results) == len(cases)
    for index, (case, result, golden) in enumerate(zip(cases, results, expected)):
        assert_json_matches(result, golden, _case_label(index, case), RELATIVE_TOLERANCE)


def test_python_reference_still_matches_goldens() -> None:
    cases, expected = _load_goldens()
    regenerated = all_cases()
    assert [json.dumps(case, sort_keys=True) for case, _ in regenerated] == [
        json.dumps(case, sort_keys=True) for case in cases
    ], "golden inputs drifted: rerun python -m tests.wildlife_tracking_parity_cases"
    for index, ((case, fresh), golden) in enumerate(zip(regenerated, expected)):
        assert_json_matches(json.loads(json.dumps(fresh)), golden, _case_label(index, case), RELATIVE_TOLERANCE)


def test_pipeline_scenarios_keep_one_entity_per_bird() -> None:
    """The scripted scenes' decision-level contract, independent of Rust."""
    cases, _ = _load_goldens()
    by_name = {case["name"]: case for case in cases if case["kind"] == "pipeline" and case["seed"] == 0}
    expected_entity_counts = {"countersinging_pair": 2, "rotating_flock": 3, "two_species_close": 2, "omni_then_localized": 1}
    for name, count in expected_entity_counts.items():
        result = run_pipeline_case(by_name[name]["steps"])
        assert len(result["entities"]) == count, name
