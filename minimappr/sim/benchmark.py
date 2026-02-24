"""Offline benchmark helpers for localization and event metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EventRecord:
    label: str
    timestamp_ns: int
    position_m: tuple[float, float, float]


def compute_event_metrics(
    truth: list[EventRecord],
    predicted: list[EventRecord],
    *,
    max_time_delta_ns: int,
    max_distance_m: float,
) -> dict[str, float | int]:
    unmatched_pred = set(range(len(predicted)))
    true_positives = 0
    false_positives = 0
    missed = 0
    confused = 0

    for event in truth:
        best_idx: int | None = None
        best_dist = float("inf")
        for idx in list(unmatched_pred):
            candidate = predicted[idx]
            if abs(candidate.timestamp_ns - event.timestamp_ns) > max_time_delta_ns:
                continue
            dist = float(np.linalg.norm(np.asarray(candidate.position_m) - np.asarray(event.position_m)))
            if dist <= max_distance_m and dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx is None:
            missed += 1
            continue

        unmatched_pred.discard(best_idx)
        true_positives += 1
        if predicted[best_idx].label.strip().lower() != event.label.strip().lower():
            confused += 1

    false_positives = len(unmatched_pred)
    confusion_rate = (confused / true_positives) if true_positives else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "missed_detections": missed,
        "class_confusions": confused,
        "confusion_rate": confusion_rate,
    }

