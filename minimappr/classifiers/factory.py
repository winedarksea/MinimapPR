"""Classifier backend selection."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from minimappr.classifiers.base import AudioClassifier
from minimappr.classifiers.chaining import ChainStage, ChainedClassifier
from minimappr.classifiers.heuristic import HeuristicClassifier
from minimappr.classifiers.yamnet import YAMNetClassifier
from minimappr.config import Settings
from minimappr.core.taxonomy import label_category_for_name


logger = logging.getLogger(__name__)


def create_classifier(settings: Settings) -> AudioClassifier:
    backend = settings.classifier_backend.strip().lower()
    base_classifier: AudioClassifier
    yamnet_active = False
    if backend == "yamnet":
        try:
            base_classifier = YAMNetClassifier(
                min_confidence=settings.yamnet_min_confidence,
                target_rms=settings.yamnet_input_target_rms,
                max_input_gain=settings.yamnet_max_input_gain,
            )
            yamnet_active = True
        except Exception as exc:  # pragma: no cover - optional runtime backend
            logger.warning("YAMNet unavailable (%s). Falling back to heuristic classifier.", exc)
            base_classifier = _create_heuristic(settings)
    elif backend == "birdnet":
        try:
            from minimappr.classifiers.birdnet import BirdNETClassifier  # noqa: PLC0415

            base_classifier = BirdNETClassifier(min_confidence=settings.birdnet_trigger_min_confidence)
        except Exception as exc:  # pragma: no cover - optional runtime backend
            logger.warning("BirdNET unavailable (%s). Falling back to heuristic classifier.", exc)
            base_classifier = _create_heuristic(settings)
    else:
        base_classifier = _create_heuristic(settings)

    stages = _load_chain_stages(settings.model_chain_config_path, settings)
    if not stages and yamnet_active:
        # When no explicit chain config exists and YAMNet is active, apply the
        # default beta chain: bird-like detections trigger BirdNET for species ID.
        stages = _default_chain_stages(settings)
    if not stages:
        return base_classifier
    return ChainedClassifier(
        base_classifier=base_classifier,
        stages=stages,
        category_for_label=label_category_for_name,
    )


def _load_chain_stages(path: Path, settings: Settings) -> list[ChainStage]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Unable to read model chain config: %s", path)
        return []
    if not isinstance(raw, dict):
        return []
    out: list[ChainStage] = []
    for item in raw.get("stages", []):
        if not isinstance(item, dict):
            continue
        stage = _build_stage(item, settings)
        if stage is not None:
            out.append(stage)
    return out


def _build_stage(raw: dict[str, Any], settings: Settings) -> ChainStage | None:
    stage_id = str(raw.get("id") or "").strip()
    backend = str(raw.get("backend") or "heuristic").strip().lower()
    if not stage_id:
        return None
    classifier: AudioClassifier
    if backend == "yamnet":
        try:
            classifier = YAMNetClassifier(
                min_confidence=settings.yamnet_min_confidence,
                target_rms=settings.yamnet_input_target_rms,
                max_input_gain=settings.yamnet_max_input_gain,
            )
        except Exception:
            return None
    elif backend == "birdnet":
        try:
            from minimappr.classifiers.birdnet import BirdNETClassifier  # noqa: PLC0415
            classifier = BirdNETClassifier(min_confidence=float(raw.get("min_confidence", 0.1) or 0.1))
        except Exception as exc:
            logger.warning("BirdNET backend unavailable (%s). Skipping chain stage '%s'.", exc, stage_id)
            return None
    else:
        classifier = _create_heuristic(settings)

    labels = _to_set(raw.get("trigger_labels"))
    categories = _to_set(raw.get("trigger_categories"))
    min_conf = float(raw.get("min_confidence", 0.0) or 0.0)
    weight = float(raw.get("score_weight", 1.0) or 1.0)
    return ChainStage(
        stage_id=stage_id,
        classifier=classifier,
        trigger_labels=labels,
        trigger_categories=categories,
        min_confidence=max(0.0, min(1.0, min_conf)),
        score_weight=max(0.0, weight),
    )


def _default_chain_stages(settings: Settings) -> list[ChainStage]:
    """Return the default beta chain when no model_chain.json config file exists.

    Triggers BirdNET species identification whenever YAMNet classifies audio
    in the 'bird' category with configurable recall-biased confidence. BirdNET is loaded
    lazily — the stage is silently omitted if the birdnet package is not installed.
    """
    try:
        from minimappr.classifiers.birdnet import BirdNETClassifier  # noqa: PLC0415
        birdnet_classifier: AudioClassifier = BirdNETClassifier(min_confidence=0.1)
    except Exception as exc:
        logger.info(
            "BirdNET package not available (%s). Default chain will omit species ID stage."
            " Install with: pip install birdnet",
            exc,
        )
        return []

    return [
        ChainStage(
            stage_id="birdnet_species",
            classifier=birdnet_classifier,
            # Trigger on YAMNet 'bird' category OR common bird labels.
            trigger_categories={"bird"},
            trigger_labels={"bird", "bird_like", "bird sound", "bird vocalization", "songbird"},
            min_confidence=settings.birdnet_trigger_min_confidence,
            score_weight=1.2,  # Prefer species-level label over coarse 'bird_like'.
        )
    ]


def _to_set(value: Any) -> set[str]:
    if isinstance(value, str):
        text = value.strip().lower()
        return {text} if text else set()
    if isinstance(value, (list, tuple, set)):
        out: set[str] = set()
        for item in value:
            text = str(item).strip().lower()
            if text:
                out.add(text)
        return out
    return set()


def _create_heuristic(settings: Settings) -> HeuristicClassifier:
    cfg = settings.classifier_config()
    return HeuristicClassifier(
        ambient_rms_threshold=cfg.heuristic_ambient_rms_threshold,
        impulse_crest_threshold=cfg.heuristic_impulse_crest_threshold,
        impulse_bandwidth_threshold_hz=cfg.heuristic_impulse_bandwidth_threshold_hz,
        bird_centroid_min_hz=cfg.heuristic_bird_centroid_min_hz,
        bird_zcr_min=cfg.heuristic_bird_zcr_min,
        speech_centroid_min_hz=cfg.heuristic_speech_centroid_min_hz,
        speech_centroid_max_hz=cfg.heuristic_speech_centroid_max_hz,
        speech_zcr_min=cfg.heuristic_speech_zcr_min,
        speech_zcr_max=cfg.heuristic_speech_zcr_max,
        speech_flatness_max=cfg.heuristic_speech_flatness_max,
        machine_centroid_max_hz=cfg.heuristic_machine_centroid_max_hz,
        machine_flatness_max=cfg.heuristic_machine_flatness_max,
        unknown_min_score=cfg.heuristic_unknown_min_score,
        unknown_score=cfg.heuristic_unknown_score,
    )
