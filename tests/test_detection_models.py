"""Detection wire formats must preserve values and honor Pydantic filtering."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from minimappr.models import DetectionEvent


@pytest.fixture
def detection_payload() -> dict[str, Any]:
    return {
        "id": "det-roundtrip",
        "timestamp_ns": 1,
        "position_m": (0.0, 0.0, 0.0),
        "confidence": 0.7,
        "gdop": 1.8,
        "label": "blue jay",
        "label_confidence": 0.8,
        "reference_sensor": "node-1:ch0",
    }


@pytest.mark.parametrize("reporting_modality", ["omni", "localized"])
@pytest.mark.parametrize("gdop", [1.8, float("inf"), None])
def test_detection_gdop_json_round_trip(detection_payload, reporting_modality, gdop) -> None:
    detection = DetectionEvent.model_validate(
        {**detection_payload, "reporting_modality": reporting_modality, "gdop": gdop}
    )
    expected_gdop = float("inf") if gdop is None else gdop
    assert detection.gdop == expected_gdop
    assert detection.model_dump()["gdop"] == expected_gdop

    dumped = detection.model_dump(mode="json")
    assert dumped["gdop"] == (None if expected_gdop == float("inf") else expected_gdop)
    restored = DetectionEvent.model_validate(dumped)
    restored_from_json = DetectionEvent.model_validate_json(detection.model_dump_json())
    assert restored == restored_from_json == detection


@pytest.mark.parametrize("field_name", ["gdop", "id", "timestamp_ns", "reference_sensor"])
def test_detection_required_fields_stay_required(detection_payload, field_name) -> None:
    del detection_payload[field_name]
    with pytest.raises(ValidationError) as error:
        DetectionEvent.model_validate(detection_payload)
    assert error.value.errors()[0]["loc"] == (field_name,)
    assert error.value.errors()[0]["type"] == "missing"


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [("gdop", "unavailable"), ("gdop", []), ("id", None), ("timestamp_ns", None)],
)
def test_detection_invalid_required_fields_raise_validation_error(
    detection_payload, field_name, invalid_value
) -> None:
    with pytest.raises(ValidationError) as error:
        DetectionEvent.model_validate({**detection_payload, field_name: invalid_value})
    assert error.value.errors()[0]["loc"] == (field_name,)


@pytest.mark.parametrize(
    ("level_fields", "expected_level"),
    [
        ({}, None),
        ({"received_level_db": -43.4}, -43.4),
        ({"spl_db": -43.4}, -43.4),
        ({"received_level_db": None}, None),
        ({"spl_db": None}, None),
        ({"received_level_db": -43.4, "spl_db": -59.0}, -43.4),
        ({"received_level_db": None, "spl_db": -43.4}, None),
        ({"received_level_db": -43.4, "spl_db": None}, -43.4),
    ],
)
def test_detection_level_legacy_input_only(detection_payload, level_fields, expected_level) -> None:
    payload = {**detection_payload, **level_fields}
    detection = DetectionEvent.model_validate(payload)
    assert detection.received_level_db == expected_level
    assert DetectionEvent.model_validate_json(json.dumps(payload)) == detection
    assert not hasattr(detection, "spl_db")
    for by_alias in (False, True):
        dumped = json.loads(detection.model_dump_json(by_alias=by_alias))
        assert dumped["received_level_db"] == expected_level
        assert "spl_db" not in dumped


@pytest.mark.parametrize("mode", ["python", "json"])
@pytest.mark.parametrize("by_alias", [False, True])
@pytest.mark.parametrize(
    ("filter_options", "expected_levels"),
    [
        ({}, {"received_level_db": -43.4}),
        ({"exclude": {"spl_db"}}, {"received_level_db": -43.4}),
        ({"include": {"received_level_db"}}, {"received_level_db": -43.4}),
        ({"include": {"spl_db"}}, {}),
        ({"exclude": {"received_level_db"}}, {}),
        ({"include": {"received_level_db", "spl_db"}}, {"received_level_db": -43.4}),
    ],
)
def test_detection_level_dump_filtering(
    detection_payload, mode, by_alias, filter_options, expected_levels
) -> None:
    detection = DetectionEvent.model_validate({**detection_payload, "spl_db": -43.4})
    dumped = detection.model_dump(mode=mode, by_alias=by_alias, **filter_options)
    if "include" in filter_options:
        assert dumped == expected_levels
    else:
        assert {key: value for key, value in dumped.items() if key.endswith("_db")} == expected_levels
    assert "spl_db" not in dumped
    if mode == "json":
        assert json.loads(detection.model_dump_json(by_alias=by_alias, **filter_options)) == dumped


@pytest.mark.parametrize("mode", ["python", "json"])
@pytest.mark.parametrize(
    ("level_fields", "filter_options", "expected_levels"),
    [
        ({}, {"exclude_none": True}, {}),
        ({"received_level_db": None}, {"exclude_none": True}, {}),
        ({}, {"exclude_unset": True}, {}),
        ({"spl_db": None}, {"exclude_unset": True}, {"received_level_db": None}),
        ({}, {"exclude_defaults": True}, {}),
        ({"spl_db": None}, {"exclude_defaults": True}, {}),
        ({"spl_db": -43.4}, {"exclude_none": True}, {"received_level_db": -43.4}),
        ({"spl_db": -43.4}, {"exclude_unset": True}, {"received_level_db": -43.4}),
        ({"spl_db": -43.4}, {"exclude_defaults": True}, {"received_level_db": -43.4}),
    ],
)
def test_detection_level_dump_omission(
    detection_payload, mode, level_fields, filter_options, expected_levels
) -> None:
    detection = DetectionEvent.model_validate({**detection_payload, **level_fields})
    dumped = detection.model_dump(mode=mode, **filter_options)
    assert {key: value for key, value in dumped.items() if key.endswith("_db")} == expected_levels
    assert "spl_db" not in dumped
