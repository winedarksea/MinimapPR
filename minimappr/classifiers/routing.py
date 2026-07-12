"""Classifier routing configuration: which classifiers run on which audio contexts.

Replaces the removed single-backend ``classifier_backend`` setting and the
linear ``model_chain.json`` chain file. A routing config names a set of
classifier *members* (``classifiers``), maps each audio *context*
(``detection_trigger`` / ``localized_render`` / ``omni_continuous``) to the
members that always run on it, attaches *chains* (stages consuming a parent
member's output — audio or in-memory embedding frames), and declares
*triggers* (side-effect hooks like speech capture, evaluated at the async
layer, never inside classifiers).

Code defaults here are authoritative; ``data/classifier_routing.json`` is a
shipped mirror an operator may edit. Settings kill switches
(``birdnet_enabled`` etc.) strip members/chains/triggers after load.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

KNOWN_BACKENDS = {"yamnet", "birdnet", "heuristic", "drone_head", "moonshine_stt"}
# Backends that consume embedding frames instead of audio: valid only as
# ``input: "embedding"`` chain stages, never in a context ``run`` list.
EMBEDDING_ONLY_BACKENDS = {"drone_head"}
# Backends whose absence drops the member with one INFO log instead of failing
# startup (optional capability, not core pipeline).
OPTIONAL_BACKENDS = {"birdnet", "moonshine_stt"}

CONTEXT_DETECTION_TRIGGER = "detection_trigger"
CONTEXT_LOCALIZED_RENDER = "localized_render"
CONTEXT_OMNI_CONTINUOUS = "omni_continuous"


@dataclass(slots=True)
class ClassifierSpec:
    member_id: str
    backend: str
    min_confidence: float = 0.0
    keep_embeddings: bool = False
    model_path: str | None = None


@dataclass(slots=True)
class ContextSpec:
    name: str
    run: tuple[str, ...] = ()
    interval_seconds: float | None = None
    window_seconds: float | None = None
    min_rms: float | None = None


@dataclass(slots=True)
class ChainSpec:
    """A stage attached after a parent member. ``chain_id`` must name an entry
    in ``classifiers`` (that spec provides the backend/model)."""

    chain_id: str
    after: str
    input: str = "audio"  # "audio" | "embedding"
    trigger_labels: frozenset[str] = frozenset()
    trigger_categories: frozenset[str] = frozenset()
    min_confidence: float = 0.0
    score_weight: float = 1.0


@dataclass(slots=True)
class TriggerSpec:
    trigger_id: str
    on: str
    action: str
    labels: frozenset[str] = frozenset()
    min_confidence: float = 0.0


@dataclass(slots=True)
class RoutingConfig:
    version: int = 1
    classifiers: dict[str, ClassifierSpec] = field(default_factory=dict)
    contexts: dict[str, ContextSpec] = field(default_factory=dict)
    chains: tuple[ChainSpec, ...] = ()
    triggers: tuple[TriggerSpec, ...] = ()

    def context(self, name: str) -> ContextSpec:
        return self.contexts.get(name) or ContextSpec(name=name)

    def chains_for_member(self, member_id: str) -> tuple[ChainSpec, ...]:
        return tuple(chain for chain in self.chains if chain.after == member_id)

    def triggers_for_action(self, action: str) -> tuple[TriggerSpec, ...]:
        return tuple(trigger for trigger in self.triggers if trigger.action == action)


def default_routing() -> RoutingConfig:
    """Authoritative code defaults (mirrored by data/classifier_routing.json)."""
    return RoutingConfig(
        version=1,
        classifiers={
            "yamnet": ClassifierSpec(
                member_id="yamnet", backend="yamnet", min_confidence=0.25, keep_embeddings=True
            ),
            "birdnet": ClassifierSpec(member_id="birdnet", backend="birdnet", min_confidence=0.40),
            "drone_head": ClassifierSpec(
                member_id="drone_head",
                backend="drone_head",
                min_confidence=0.50,
                model_path="data/models/drone_head.onnx",
            ),
            "stt": ClassifierSpec(member_id="stt", backend="moonshine_stt"),
        },
        contexts={
            CONTEXT_DETECTION_TRIGGER: ContextSpec(
                name=CONTEXT_DETECTION_TRIGGER, run=("yamnet", "birdnet")
            ),
            CONTEXT_LOCALIZED_RENDER: ContextSpec(
                name=CONTEXT_LOCALIZED_RENDER, run=("yamnet", "birdnet")
            ),
            CONTEXT_OMNI_CONTINUOUS: ContextSpec(
                name=CONTEXT_OMNI_CONTINUOUS,
                run=("birdnet",),
                interval_seconds=30.0,
                window_seconds=15.0,
                min_rms=0.001,
            ),
        },
        chains=(
            ChainSpec(chain_id="drone_head", after="yamnet", input="embedding"),
        ),
        triggers=(
            TriggerSpec(
                trigger_id="stt_on_speech",
                on="yamnet",
                action="speech_capture",
                labels=frozenset({"speech"}),
                min_confidence=0.50,
            ),
        ),
    )


def _to_label_set(value: Any) -> frozenset[str]:
    if isinstance(value, str):
        text = value.strip().lower()
        return frozenset({text}) if text else frozenset()
    if isinstance(value, (list, tuple, set)):
        return frozenset(str(item).strip().lower() for item in value if str(item).strip())
    return frozenset()


def _parse_routing(raw: dict[str, Any], source: str) -> RoutingConfig:
    classifiers: dict[str, ClassifierSpec] = {}
    for member_id, spec in (raw.get("classifiers") or {}).items():
        if not isinstance(spec, dict):
            raise ValueError(f"{source}: classifiers[{member_id!r}] must be an object")
        backend = str(spec.get("backend") or "").strip().lower()
        if backend not in KNOWN_BACKENDS:
            raise ValueError(
                f"{source}: classifiers[{member_id!r}].backend {backend!r} "
                f"is not one of {sorted(KNOWN_BACKENDS)}"
            )
        classifiers[str(member_id)] = ClassifierSpec(
            member_id=str(member_id),
            backend=backend,
            min_confidence=float(spec.get("min_confidence", 0.0) or 0.0),
            keep_embeddings=bool(spec.get("keep_embeddings", False)),
            model_path=(str(spec["model_path"]) if spec.get("model_path") else None),
        )

    contexts: dict[str, ContextSpec] = {}
    for name, spec in (raw.get("contexts") or {}).items():
        if not isinstance(spec, dict):
            raise ValueError(f"{source}: contexts[{name!r}] must be an object")
        run = spec.get("run", [])
        if not isinstance(run, list):
            raise ValueError(f"{source}: contexts[{name!r}].run must be a list")
        for member_id in run:
            member = classifiers.get(str(member_id))
            if member is None:
                raise ValueError(
                    f"{source}: contexts[{name!r}].run references unknown classifier {member_id!r}"
                )
            if member.backend in EMBEDDING_ONLY_BACKENDS:
                raise ValueError(
                    f"{source}: classifier {member_id!r} ({member.backend}) is embedding-only "
                    f"and cannot appear in contexts[{name!r}].run — attach it as a chain with "
                    'input: "embedding"'
                )
        contexts[str(name)] = ContextSpec(
            name=str(name),
            run=tuple(str(member_id) for member_id in run),
            interval_seconds=(
                float(spec["interval_seconds"]) if spec.get("interval_seconds") is not None else None
            ),
            window_seconds=(
                float(spec["window_seconds"]) if spec.get("window_seconds") is not None else None
            ),
            min_rms=(float(spec["min_rms"]) if spec.get("min_rms") is not None else None),
        )

    chains: list[ChainSpec] = []
    for item in raw.get("chains") or []:
        if not isinstance(item, dict):
            raise ValueError(f"{source}: chains entries must be objects")
        chain_id = str(item.get("id") or "").strip()
        if chain_id not in classifiers:
            raise ValueError(
                f"{source}: chain {chain_id!r} must name an entry in classifiers"
            )
        input_kind = str(item.get("input") or "audio").strip().lower()
        if input_kind not in {"audio", "embedding"}:
            raise ValueError(f"{source}: chain {chain_id!r} input must be 'audio' or 'embedding'")
        if classifiers[chain_id].backend in EMBEDDING_ONLY_BACKENDS and input_kind != "embedding":
            raise ValueError(
                f"{source}: chain {chain_id!r} backend is embedding-only; input must be 'embedding'"
            )
        chains.append(
            ChainSpec(
                chain_id=chain_id,
                after=str(item.get("after") or "").strip(),
                input=input_kind,
                trigger_labels=_to_label_set(item.get("trigger_labels")),
                trigger_categories=_to_label_set(item.get("trigger_categories")),
                min_confidence=float(item.get("min_confidence", 0.0) or 0.0),
                score_weight=float(item.get("score_weight", 1.0) or 1.0),
            )
        )

    triggers: list[TriggerSpec] = []
    for item in raw.get("triggers") or []:
        if not isinstance(item, dict):
            raise ValueError(f"{source}: triggers entries must be objects")
        triggers.append(
            TriggerSpec(
                trigger_id=str(item.get("id") or "").strip(),
                on=str(item.get("on") or "").strip(),
                action=str(item.get("action") or "").strip(),
                labels=_to_label_set(item.get("labels")),
                min_confidence=float(item.get("min_confidence", 0.0) or 0.0),
            )
        )

    return RoutingConfig(
        version=int(raw.get("version", 1)),
        classifiers=classifiers,
        contexts=contexts,
        chains=tuple(chains),
        triggers=tuple(triggers),
    )


def load_routing_file(path: Path) -> RoutingConfig:
    """Load a routing config from ``path``; missing file yields code defaults.

    A malformed file raises: silently substituting defaults would change which
    models run on live audio.
    """
    if not path.exists():
        return default_routing()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to parse classifier routing config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Classifier routing config {path} must be a JSON object")
    return _parse_routing(raw, source=str(path))


def apply_settings(routing: RoutingConfig, settings: Any) -> RoutingConfig:
    """Overlay Settings kill switches and threshold overrides onto ``routing``.

    ``birdnet_enabled=False`` strips birdnet members from every context;
    ``drone_head_enabled=False`` strips drone_head chains; ``stt_enabled=False``
    strips ``speech_capture`` triggers. Threshold settings override the matching
    spec fields so the PATCHable knobs stay live without editing the JSON.
    """
    classifiers = dict(routing.classifiers)

    def _member_disabled(member_id: str) -> bool:
        spec = classifiers.get(member_id)
        if spec is None:
            return True
        if spec.backend == "birdnet" and not getattr(settings, "birdnet_enabled", True):
            return True
        return False

    contexts = {
        name: replace(
            ctx, run=tuple(member for member in ctx.run if not _member_disabled(member))
        )
        for name, ctx in routing.contexts.items()
    }

    chains = tuple(
        chain
        for chain in routing.chains
        if not (
            classifiers.get(chain.chain_id) is not None
            and classifiers[chain.chain_id].backend == "drone_head"
            and not getattr(settings, "drone_head_enabled", True)
        )
        and not _member_disabled(chain.chain_id)
    )

    triggers = tuple(
        trigger
        for trigger in routing.triggers
        if not (trigger.action == "speech_capture" and not getattr(settings, "stt_enabled", True))
    )

    for member_id, spec in list(classifiers.items()):
        if spec.backend == "drone_head":
            classifiers[member_id] = replace(
                spec,
                min_confidence=float(
                    getattr(settings, "drone_head_min_confidence", spec.min_confidence)
                ),
                model_path=str(
                    getattr(settings, "drone_head_model_path", spec.model_path or "")
                )
                or spec.model_path,
            )

    triggers = tuple(
        replace(
            trigger,
            min_confidence=float(
                getattr(settings, "stt_trigger_min_confidence", trigger.min_confidence)
            ),
        )
        if trigger.action == "speech_capture"
        else trigger
        for trigger in triggers
    )

    omni = contexts.get(CONTEXT_OMNI_CONTINUOUS)
    if omni is not None:
        contexts[CONTEXT_OMNI_CONTINUOUS] = replace(
            omni,
            interval_seconds=float(
                getattr(settings, "omni_scan_interval_seconds", omni.interval_seconds or 30.0)
            ),
            window_seconds=float(
                getattr(settings, "omni_scan_window_seconds", omni.window_seconds or 15.0)
            ),
            min_rms=float(getattr(settings, "omni_scan_min_rms", omni.min_rms or 0.0)),
        )

    return RoutingConfig(
        version=routing.version,
        classifiers=classifiers,
        contexts=contexts,
        chains=chains,
        triggers=triggers,
    )


def load_routing(settings: Any) -> RoutingConfig:
    """Load the routing config named by settings and apply its kill switches."""
    path = Path(getattr(settings, "classifier_routing_config_path", "data/classifier_routing.json"))
    return apply_settings(load_routing_file(path), settings)


__all__ = [
    "CONTEXT_DETECTION_TRIGGER",
    "CONTEXT_LOCALIZED_RENDER",
    "CONTEXT_OMNI_CONTINUOUS",
    "ChainSpec",
    "ClassifierSpec",
    "ContextSpec",
    "EMBEDDING_ONLY_BACKENDS",
    "KNOWN_BACKENDS",
    "OPTIONAL_BACKENDS",
    "RoutingConfig",
    "TriggerSpec",
    "apply_settings",
    "default_routing",
    "load_routing",
    "load_routing_file",
]
