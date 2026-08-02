"""Build per-context classifiers from the routing config.

``create_context_classifier(settings, context)`` resolves the routing config's
``run`` list for that context into a CompositeClassifier (unwrapped when only
one member survives), attaching chain stages to their parent members.

Failure policy (mirrors routing.OPTIONAL_BACKENDS): a missing optional backend
(BirdNET, Moonshine) drops the member with one INFO log; a YAMNet build
failure is a hard startup error — YAMNet is bundled and load-bearing (drone
head chain, speech triggers). A missing drone-head model file drops the chain
with a WARNING (the artifact ships separately from the code).
"""

from __future__ import annotations

import logging
import inspect
from pathlib import Path

from minimappr.classifiers.base import AudioClassifier
from minimappr.classifiers.chaining import ChainStage, ChainedClassifier
from minimappr.classifiers.composite import CompositeClassifier, CompositeMember
from minimappr.classifiers.heuristic import HeuristicClassifier
from minimappr.classifiers.preprocessed import PreprocessedClassifier
from minimappr.classifiers.routing import (
    CONTEXT_DETECTION_TRIGGER,
    ChainSpec,
    ClassifierSpec,
    RoutingConfig,
    load_routing,
)
from minimappr.config import Settings
from minimappr.audio_processing.chain import build_preprocessing_chain
from minimappr.audio_processing.profiles import (
    load_audio_processing_configuration,
    profile_fingerprint,
)
from minimappr.core.taxonomy import label_category_for_name

try:
    from minimappr.classifiers.yamnet import YAMNetClassifier
except Exception:
    YAMNetClassifier = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


def create_classifier(settings: Settings) -> AudioClassifier:
    """Classifier for the detection-trigger pipeline (back-compat entry point)."""
    return create_context_classifier(settings, CONTEXT_DETECTION_TRIGGER)


def create_context_classifier(
    settings: Settings,
    context: str,
    routing: RoutingConfig | None = None,
) -> AudioClassifier:
    if routing is None:
        routing = load_routing(settings)
    _warn_on_legacy_chain_config(settings)

    context_spec = routing.context(context)
    members: list[CompositeMember] = []
    for member_id in context_spec.run:
        spec = routing.classifiers.get(member_id)
        if spec is None:
            continue
        classifier = _build_member(spec, settings, routing.chains_for_member(member_id), routing)
        if classifier is None:
            continue
        members.append(CompositeMember(member_id=member_id, classifier=classifier))

    if not members:
        logger.info(
            "Classifier routing resolved zero members for context %r; using heuristic.",
            context,
        )
        return _create_heuristic(settings)
    if len(members) == 1:
        return members[0].classifier
    # Members may advertise safety-critical labels (e.g. the T3/T4 alarm
    # detector) that must not be masked by a higher-scoring sibling.
    priority_labels: frozenset[str] = frozenset()
    for member in members:
        priority_labels |= getattr(member.classifier, "PRIORITY_LABELS", frozenset())
    return CompositeClassifier(members, priority_labels=priority_labels)


def _build_member(
    spec: ClassifierSpec,
    settings: Settings,
    chains: tuple[ChainSpec, ...],
    routing: RoutingConfig,
) -> AudioClassifier | None:
    base = _build_backend(spec, settings, needs_embeddings=any(c.input == "embedding" for c in chains))
    if base is None:
        return None
    base = _apply_member_preprocessor(base, spec, settings)
    stages = [
        stage
        for stage in (_build_chain_stage(chain, settings, routing) for chain in chains)
        if stage is not None
    ]
    if not stages:
        return base
    return ChainedClassifier(
        base_classifier=base,
        stages=stages,
        category_for_label=label_category_for_name,
    )


def _build_backend(
    spec: ClassifierSpec,
    settings: Settings,
    *,
    needs_embeddings: bool = False,
) -> AudioClassifier | None:
    backend = spec.backend
    if backend == "yamnet":
        if YAMNetClassifier is None:
            raise RuntimeError(
                "YAMNet classifier is required by the routing config but its dependencies "
                "failed to import. Install the core dependencies "
                "(pip install minimappr) or remove 'yamnet' from the routing config."
            )
        preprocess_profile = None
        if spec.preprocess_profile:
            preprocess_profile = load_audio_processing_configuration(
                settings.audio_processing_config_path
            ).profile(spec.preprocess_profile)
        constructor_kwargs = {
            "min_confidence": spec.min_confidence or settings.yamnet_min_confidence,
            "target_rms": settings.yamnet_input_target_rms,
            "max_input_gain": settings.yamnet_max_input_gain,
            "keep_embeddings": spec.keep_embeddings or needs_embeddings,
        }
        # Signature probing keeps lightweight test and downstream compatibility
        # stubs usable while the concrete backend adopts profile binding.
        if "preprocess_profile" in inspect.signature(YAMNetClassifier).parameters:
            constructor_kwargs["preprocess_profile"] = preprocess_profile
        return YAMNetClassifier(**constructor_kwargs)
    if backend == "birdnet":
        try:
            from minimappr.classifiers.birdnet import BirdNETClassifier  # noqa: PLC0415

            birdnet_kwargs = {
                "min_confidence": spec.min_confidence or settings.birdnet_trigger_min_confidence,
                "latitude": settings.site_origin_lat,
                "longitude": settings.site_origin_lon,
                "geo_min_confidence": settings.birdnet_geo_min_confidence,
            }
            # pool_size=1 is a de-facto lock: with N fusion workers, N-1 park
            # in queue.get() for the whole inference (the 2026-08-01 live-box
            # serialization root cause). Each extra session costs ~125 MB + one
            # TFLite subprocess, so the default stays 1. Signature-probed so
            # lightweight test/downstream stubs stay usable.
            birdnet_parameters = inspect.signature(BirdNETClassifier).parameters
            accepts_pool_size = "pool_size" in birdnet_parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in birdnet_parameters.values()
            )
            if accepts_pool_size:
                birdnet_kwargs["pool_size"] = max(
                    1, int(getattr(settings, "birdnet_pool_size", 1))
                )
            # Request coalescing. A run_arrays() call costs ~1.0s of fixed
            # barrier synchronization whatever the payload, so batching
            # concurrent callers into one call is worth 3-4x. Same signature
            # probe as pool_size so stubs stay usable.
            if "batch_max_wait_seconds" in birdnet_parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in birdnet_parameters.values()
            ):
                birdnet_kwargs["batch_max_wait_seconds"] = float(
                    getattr(settings, "birdnet_batch_max_wait_seconds", 0.0)
                )
                birdnet_kwargs["batch_max_size"] = int(
                    getattr(settings, "birdnet_batch_max_size", 16)
                )
            return BirdNETClassifier(**birdnet_kwargs)
        except Exception as exc:  # pragma: no cover - optional runtime backend
            logger.info(
                "BirdNET unavailable (%s); dropping routing member %r. Install with: pip install birdnet",
                exc,
                spec.member_id,
            )
            return None
    if backend == "drone_head":
        try:
            from minimappr.classifiers.drone_head import (  # noqa: PLC0415
                DroneHeadClassifier,
                PreprocessingProfileMismatchError,
            )
            yamnet_profile = load_audio_processing_configuration(
                settings.audio_processing_config_path
            ).profile("yamnet")

            return DroneHeadClassifier(
                model_path=spec.model_path or settings.drone_head_model_path,
                min_confidence=spec.min_confidence,
                min_frame_fraction=(
                    spec.min_frame_fraction if spec.min_frame_fraction is not None else 0.2
                ),
                ambient_margin=(
                    spec.ambient_margin if spec.ambient_margin is not None else 0.0
                ),
                min_mean_confidence=(
                    spec.min_mean_confidence if spec.min_mean_confidence is not None else 0.0
                ),
                expected_preprocessing_fingerprint=profile_fingerprint(yamnet_profile),
            )
        except PreprocessingProfileMismatchError:
            raise
        except FileNotFoundError as exc:
            logger.warning("Drone head model missing; dropping member %r: %s", spec.member_id, exc)
            return None
        except Exception as exc:
            logger.warning("Drone head unavailable (%s); dropping member %r.", exc, spec.member_id)
            return None
    if backend == "t3t4_alarm":
        from minimappr.classifiers.temporal_alarm import TemporalAlarmClassifier  # noqa: PLC0415

        return TemporalAlarmClassifier(
            min_repeats=settings.t3t4_min_repeats,
            min_confidence=spec.min_confidence or settings.t3t4_min_confidence,
            tone_band_low_hz=settings.t3t4_tone_band_low_hz,
            tone_band_high_hz=settings.t3t4_tone_band_high_hz,
            tolerance=settings.t3t4_tolerance,
            hysteresis_hi_ratio=settings.t3t4_hysteresis_hi_ratio,
            hysteresis_lo_ratio=settings.t3t4_hysteresis_lo_ratio,
        )
    if backend == "heuristic":
        return _create_heuristic(settings)
    logger.warning("Unknown classifier backend %r; dropping member %r.", backend, spec.member_id)
    return None


def _build_chain_stage(
    chain: ChainSpec,
    settings: Settings,
    routing: RoutingConfig,
) -> ChainStage | None:
    spec = routing.classifiers.get(chain.chain_id)
    if spec is None:
        return None
    classifier = _build_backend(spec, settings)
    if classifier is None:
        return None
    if chain.input == "audio":
        classifier = _apply_member_preprocessor(classifier, spec, settings)
    return ChainStage(
        stage_id=chain.chain_id,
        classifier=classifier,
        trigger_labels=set(chain.trigger_labels),
        trigger_categories=set(chain.trigger_categories),
        min_confidence=max(0.0, min(1.0, chain.min_confidence)),
        score_weight=max(0.0, chain.score_weight),
        input_kind=chain.input,
    )


def _apply_member_preprocessor(
    classifier: AudioClassifier,
    spec: ClassifierSpec,
    settings: Settings,
) -> AudioClassifier:
    if not spec.preprocess_profile or spec.backend == "yamnet":
        return classifier
    processing = load_audio_processing_configuration(settings.audio_processing_config_path)
    profile = processing.profile(spec.preprocess_profile)
    return PreprocessedClassifier(
        classifier=classifier,
        preprocessor=build_preprocessing_chain([dict(stage) for stage in profile.stages]),
        profile_name=profile.name,
    )


_legacy_chain_warned = False


def _warn_on_legacy_chain_config(settings: Settings) -> None:
    global _legacy_chain_warned
    if _legacy_chain_warned:
        return
    _legacy_chain_warned = True
    legacy_path = Path("data/model_chain.json")
    if legacy_path.exists():
        logger.error(
            "model_chain.json is no longer read (%s). Classifier chains now live in "
            "the routing config (%s) — migrate your stages there and delete the old file.",
            legacy_path,
            getattr(settings, "classifier_routing_config_path", "data/classifier_routing.json"),
        )


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


__all__ = ["create_classifier", "create_context_classifier"]
