"""WAV dataset discovery for the drone-head trainer.

All training audio is WAV. This module walks a ``wav_root/<label>/`` drop-dir
convention plus the server's per-detection promoted-example directory
(``data/training/{id}.json`` + ``.wav`` / ``.npy``), returning :class:`WavExample`
records. Long recordings are chopped into overlapping segments so a handful of
long clips (e.g. the 4 coyote recordings) yield many training windows.

Group keys keep segments and augmentation copies of one source file in the same
train/val/test fold so near-duplicate frames never leak across folds.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import wave
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Timestamp-minute group key from a "YYYY-MM-DD_HH-MM-SS" style stem (legacy
# recordings where consecutive clips are near-duplicates).
_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[_T](\d{2})[-:](\d{2})")

# Default aliases mapping raw promoted labels onto the canonical label set.
DEFAULT_LABEL_ALIASES = {"no_drone": "ambient", "no drone": "ambient"}


@dataclass(frozen=True)
class WavExample:
    """One training window sourced from a WAV file (or promoted npy embedding)."""

    path: Path
    label: str
    group: str
    offset_s: float
    duration_s: float
    # origin: wav | promoted | promoted_npy
    origin: str


def group_key_for(path: Path) -> str:
    """Stable per-source grouping key.

    Uses a timestamp-minute key when the stem looks like a legacy recording,
    else ``sha1(relative-or-name)[:12]`` so every physical source file gets its
    own group (segments / aug copies of one WAV never straddle folds).
    """
    m = _TS_RE.search(path.stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}-{m.group(3)}"
    return hashlib.sha1(str(path.name).encode()).hexdigest()[:12]


def _wav_duration_seconds(path: Path) -> float | None:
    """Return WAV duration in seconds via the stdlib (no soundfile dep)."""
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except (wave.Error, OSError, EOFError) as exc:
        logger.warning("Skipping unreadable WAV %s: %s", path, exc)
        return None


def _segment_offsets(
    duration_s: float, segment_seconds: float, segment_overlap: float
) -> list[tuple[float, float]]:
    """Return ``(offset_s, seg_duration_s)`` windows covering ``duration_s``.

    Short clips (<= segment_seconds) yield a single full-length window. Longer
    clips are chopped into overlapping windows; the final partial window is kept
    only if it retains at least half a segment of audio.
    """
    if duration_s <= 0:
        return []
    if duration_s <= segment_seconds:
        return [(0.0, duration_s)]
    step = max(segment_seconds * (1.0 - segment_overlap), 1e-3)
    offsets: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_s:
        remaining = duration_s - start
        if remaining < segment_seconds:
            # Keep a trailing window only if it is at least half a segment.
            if remaining >= segment_seconds * 0.5:
                offsets.append((start, remaining))
            break
        offsets.append((start, segment_seconds))
        start += step
    return offsets


def discover_wavs(
    wav_root: Path,
    labels: list[str],
    *,
    segment_seconds: float = 4.0,
    segment_overlap: float = 0.5,
) -> list[WavExample]:
    """Scan ``wav_root/<label>/**/*.wav`` into segmented :class:`WavExample` records.

    Directories whose name is not in ``labels`` are warned about and skipped.
    """
    wav_root = Path(wav_root)
    label_set = {label.strip().lower() for label in labels}
    examples: list[WavExample] = []
    if not wav_root.exists():
        logger.warning("WAV root %s does not exist; no WAV examples", wav_root)
        return examples

    for child in sorted(wav_root.iterdir()):
        if not child.is_dir():
            continue
        label = child.name.strip().lower()
        if label not in label_set:
            logger.warning("Unknown label dir %s (not in %s); skipping", child, labels)
            continue
        files = sorted(child.rglob("*.wav"))
        logger.info("%s: %d WAV files", label, len(files))
        for path in files:
            duration = _wav_duration_seconds(path)
            if duration is None:
                continue
            group = group_key_for(path)
            for offset_s, seg_dur in _segment_offsets(
                duration, segment_seconds, segment_overlap
            ):
                examples.append(
                    WavExample(
                        path=path,
                        label=label,
                        group=group,
                        offset_s=offset_s,
                        duration_s=seg_dur,
                        origin="wav",
                    )
                )
    return examples


def load_promoted(
    promoted_dir: Path,
    labels: list[str],
    label_aliases: dict[str, str] | None = None,
) -> list[WavExample]:
    """Load per-detection promoted examples written by the server.

    Each ``{id}.json`` manifest carries ``training.label`` and
    ``training.example_kind``. ``example_kind == "negative"`` maps to the
    negative label (``ambient``); aliases remap raw labels (``no_drone`` ->
    ``ambient``). A sibling ``{id}.wav`` is preferred (frame-level embedding);
    the ``{id}.npy`` mean-pooled vector is used only when the WAV is missing.
    """
    promoted_dir = Path(promoted_dir)
    if not promoted_dir.exists():
        logger.info("No promoted dir at %s; skipping promoted examples", promoted_dir)
        return []

    aliases = dict(DEFAULT_LABEL_ALIASES)
    aliases.update({k.strip().lower(): v.strip().lower() for k, v in (label_aliases or {}).items()})
    label_set = {label.strip().lower() for label in labels}
    negative_label = "ambient" if "ambient" in label_set else next(iter(labels)).strip().lower()

    examples: list[WavExample] = []
    for manifest_path in sorted(promoted_dir.glob("*.json")):
        if manifest_path.name.endswith(".tmp"):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable promoted manifest %s: %s", manifest_path, exc)
            continue

        training = manifest.get("training") or {}
        example_kind = str(training.get("example_kind") or "").strip().lower()
        raw_label = str(training.get("label") or "").strip().lower()
        if example_kind == "negative":
            label = negative_label
        else:
            label = aliases.get(raw_label, raw_label)
        if label not in label_set:
            logger.warning(
                "Promoted %s label %r not in %s; skipping", manifest_path.name, label, labels
            )
            continue

        example_id = manifest_path.stem
        wav_path = manifest_path.with_suffix(".wav")
        npy_path = manifest_path.with_suffix(".npy")
        group = f"promoted:{example_id}"
        if wav_path.exists():
            examples.append(
                WavExample(
                    path=wav_path,
                    label=label,
                    group=group,
                    offset_s=0.0,
                    duration_s=0.0,  # 0 => whole file
                    origin="promoted",
                )
            )
        elif npy_path.exists():
            examples.append(
                WavExample(
                    path=npy_path,
                    label=label,
                    group=group,
                    offset_s=0.0,
                    duration_s=0.0,
                    origin="promoted_npy",
                )
            )
        else:
            logger.warning("Promoted %s has neither .wav nor .npy; skipping", example_id)
    logger.info("Loaded %d promoted examples from %s", len(examples), promoted_dir)
    return examples


__all__ = ["WavExample", "discover_wavs", "group_key_for", "load_promoted", "DEFAULT_LABEL_ALIASES"]
