"""Claim management for filesystem ingest spool consumption."""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any, cast

from minimappr.api.journal_reader import JournalPayloadHandle
from minimappr.api.journal_state import JournalConsumerCursor, JournalConsumerStateStore
from minimappr.api.rust_dsp_manifests import JournalCursorUpdate, _lookup_cursor_update, load_localized_render_manifest_bundle, manifest_source_key
from minimappr.api.spool_consumer_types import ClaimedIngestItem, ClaimedLocalizedNodeHeartbeatItem, ClaimedLocalizedRenderItem
from minimappr.models import NodeSpec


def _localized_manifest_item_is_cursor_covered(
    item: ClaimedLocalizedRenderItem | ClaimedLocalizedNodeHeartbeatItem,
    journal_consumer_state_path: Path,
    consumer_name: str,
) -> bool:
    if not item.cursor_updates:
        return False
    state_store = JournalConsumerStateStore(journal_consumer_state_path)
    cursor_by_stream = state_store.load_cursors(consumer_name)
    return all(
        _cursor_covers_entry(
            cursor_by_stream.get(cursor_update.stream_key),
            cursor_update.journal_epoch,
            cursor_update.journal_sequence,
        )
        for cursor_update in item.cursor_updates
    )


def _list_ready_manifests_oldest_first(ready_dir: Path) -> list[Path]:
    manifests = [path for path in ready_dir.glob("*.json") if path.is_file()]

    def _sort_key(path: Path) -> str:
        stem = path.stem
        if stem and stem[0].isdigit():
            return stem
        try:
            return str(path.stat().st_mtime_ns)
        except Exception:  # noqa: BLE001
            return stem

    return sorted(manifests, key=_sort_key)


def _claim_manifest(manifest_path: Path, processing_dir: Path) -> Path | None:
    processing_dir.mkdir(parents=True, exist_ok=True)
    destination = processing_dir / manifest_path.name
    try:
        manifest_path.replace(destination)
    except FileNotFoundError:
        return None
    return destination


def _load_claimed_spool_item(manifest_path: Path) -> ClaimedIngestItem:
    with manifest_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    body_filename = str(metadata["body_filename"])
    if Path(body_filename).name != body_filename:
        raise ValueError(f"Invalid spool body filename {body_filename!r}")
    body_path = manifest_path.parent / body_filename
    ready_body_path = manifest_path.parent.parent / "ready" / body_filename
    if not body_path.exists() and ready_body_path.exists():
        ready_body_path.replace(body_path)
    if not body_path.is_file():
        raise FileNotFoundError(f"Spool body file missing for {manifest_path.name}: {body_filename}")
    return ClaimedIngestItem(
        ingest_id=str(metadata.get("spool_id") or manifest_path.stem),
        source_kind="spool",
        endpoint=str(metadata["endpoint"]),
        received_ns=int(metadata["received_ns"]),
        body_path=body_path,
        cleanup_paths=(body_path, manifest_path),
    )


def _load_claimed_journal_item(claim_path: Path, journal_streams_dir: Path) -> ClaimedIngestItem:
    with claim_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if str(metadata.get("source_kind", "")).strip().lower() != "journal":
        raise ValueError(f"Invalid journal claim file {claim_path.name}: missing source_kind=journal")

    journal_id = str(metadata.get("journal_id") or _journal_id_from_claim_path(claim_path))
    stream_key = str(metadata["stream_key"])
    if Path(stream_key).name != stream_key:
        raise ValueError(f"Invalid journal stream key {stream_key!r}")

    segment_id = str(metadata["segment_id"])
    if Path(segment_id).name != segment_id:
        raise ValueError(f"Invalid journal segment id {segment_id!r}")

    journal_epoch = int(metadata["journal_epoch"])
    journal_sequence = int(metadata["journal_sequence"])
    body_offset_bytes = int(metadata["body_offset_bytes"])
    body_length_bytes = int(metadata["body_length_bytes"])
    if body_offset_bytes < 0 or body_length_bytes < 0:
        raise ValueError(f"Journal entry {journal_id} has invalid byte range")

    segment_path = journal_streams_dir / stream_key / "segments" / f"{segment_id}.bin"
    if not segment_path.is_file():
        raise FileNotFoundError(
            f"Journal segment file missing for {claim_path.name}: {segment_path.name}"
        )

    return ClaimedIngestItem(
        ingest_id=journal_id,
        source_kind="journal",
        endpoint=str(metadata["endpoint"]),
        received_ns=int(metadata.get("ingest_received_ns") or metadata["received_ns"]),
        cleanup_paths=(claim_path,),
        stream_key=stream_key,
        journal_epoch=journal_epoch,
        journal_sequence=journal_sequence,
        segment_path=segment_path,
        body_offset_bytes=body_offset_bytes,
        body_length_bytes=body_length_bytes,
        integrity_hash=str(metadata.get("integrity_hash") or ""),
        sample_index_start=_optional_int(metadata.get("sample_index_start")),
        sample_count=_optional_int(metadata.get("sample_count")),
    )


def _delete_item_files(paths: tuple[Path, ...]) -> None:
    for path in paths:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def _complete_claimed_item(
    item: ClaimedIngestItem | ClaimedLocalizedRenderItem | ClaimedLocalizedNodeHeartbeatItem,
    journal_consumer_state_path: Path,
    consumer_name: str,
    status: str,
) -> None:
    if isinstance(item, (ClaimedLocalizedRenderItem, ClaimedLocalizedNodeHeartbeatItem)):
        state_store = JournalConsumerStateStore(journal_consumer_state_path)
        handled_ns = time.time_ns()
        for cursor_update in item.cursor_updates:
            state_store.mark_handled(
                consumer_name=consumer_name,
                stream_key=cursor_update.stream_key,
                journal_epoch=cursor_update.journal_epoch,
                journal_sequence=cursor_update.journal_sequence,
                journal_id=cursor_update.journal_id,
                status=status,
                handled_ns=handled_ns,
            )
        _delete_item_files(item.cleanup_paths)
        return
    if item.source_kind == "journal":
        if item.stream_key is None or item.journal_epoch is None or item.journal_sequence is None:
            raise ValueError(f"Journal item {item.ingest_id} is missing durable cursor metadata")
        JournalConsumerStateStore(journal_consumer_state_path).mark_handled(
            consumer_name=consumer_name,
            stream_key=item.stream_key,
            journal_epoch=item.journal_epoch,
            journal_sequence=item.journal_sequence,
            journal_id=item.ingest_id,
            status=status,
            handled_ns=time.time_ns(),
        )
    _delete_item_files(item.cleanup_paths)


def _move_claimed_item_to_failed(
    manifest_path: Path,
    failed_dir: Path,
    journal_consumer_state_path: Path,
    consumer_name: str,
    detail: str,
) -> None:
    failed_dir.mkdir(parents=True, exist_ok=True)
    body_path: Path | None = None
    with contextlib.suppress(Exception):
        with manifest_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if str(metadata.get("source_kind", "")).strip().lower() == "journal_localized_render_manifest":
            state_store = JournalConsumerStateStore(journal_consumer_state_path)
            handled_ns = time.time_ns()
            for payload in metadata.get("cursor_updates", []):
                state_store.mark_handled(
                    consumer_name=consumer_name,
                    stream_key=str(payload["stream_key"]),
                    journal_epoch=int(payload["journal_epoch"]),
                    journal_sequence=int(payload["journal_sequence"]),
                    journal_id=str(payload["journal_id"]),
                    status="failed",
                    handled_ns=handled_ns,
                    detail=detail,
                )
            for filename in metadata.get("cleanup_filenames", []):
                path = manifest_path.parent / str(filename)
                if path.exists():
                    path.replace(failed_dir / path.name)
            if manifest_path.exists():
                manifest_path.replace(failed_dir / manifest_path.name)
            return
        if str(metadata.get("source_kind", "")).strip().lower() == "journal":
            JournalConsumerStateStore(journal_consumer_state_path).mark_handled(
                consumer_name=consumer_name,
                stream_key=str(metadata["stream_key"]),
                journal_epoch=int(metadata["journal_epoch"]),
                journal_sequence=int(metadata["journal_sequence"]),
                journal_id=str(metadata.get("journal_id") or _journal_id_from_claim_path(manifest_path)),
                status="failed",
                handled_ns=time.time_ns(),
                detail=detail,
            )
            if manifest_path.exists():
                manifest_path.replace(failed_dir / manifest_path.name)
            return
        body_filename = str(metadata["body_filename"])
        if Path(body_filename).name == body_filename:
            body_path = manifest_path.parent / body_filename
    if body_path is not None and body_path.exists():
        body_path.replace(failed_dir / body_path.name)
    if manifest_path.exists():
        manifest_path.replace(failed_dir / manifest_path.name)


def _claim_journal_entries_oldest_first(
    journal_streams_dir: Path,
    processing_dir: Path,
    journal_consumer_state_path: Path,
    consumer_name: str,
) -> list[Path]:
    if not journal_streams_dir.exists():
        return []

    processing_dir.mkdir(parents=True, exist_ok=True)
    state_store = JournalConsumerStateStore(journal_consumer_state_path)
    state_store.ensure_initialized()

    cursor_by_stream = state_store.load_cursors(consumer_name)
    exception_ids = state_store.load_exception_ids(consumer_name)
    active_stream_keys = _processing_stream_keys(processing_dir)

    entries: list[tuple[tuple[int, str, int, int], dict[str, Any]]] = []
    for index_path in sorted(journal_streams_dir.glob("*/segments/*.index.jsonl")):
        default_stream_key = index_path.parent.parent.name
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw_entry = json.loads(line)
                stream_key = str(raw_entry.get("stream_key") or default_stream_key)
                journal_sequence = int(raw_entry["journal_sequence"])
                journal_epoch = int(raw_entry.get("journal_epoch") or 0)
                journal_id = str(raw_entry["journal_id"])
                if _cursor_covers_entry(cursor_by_stream.get(stream_key), journal_epoch, journal_sequence):
                    continue
                if journal_id in exception_ids:
                    continue
                entry = {
                    "source_kind": "journal",
                    "journal_id": journal_id,
                    "journal_epoch": journal_epoch,
                    "journal_sequence": journal_sequence,
                    "stream_key": stream_key,
                    "segment_id": str(raw_entry["segment_id"]),
                    "endpoint": str(raw_entry["endpoint"]),
                    "received_ns": int(raw_entry.get("ingest_received_ns") or raw_entry["received_ns"]),
                    "body_offset_bytes": int(raw_entry.get("payload_offset_bytes") or raw_entry["body_offset_bytes"]),
                    "body_length_bytes": int(raw_entry.get("payload_length_bytes") or raw_entry["body_length_bytes"]),
                    "integrity_hash": str(raw_entry.get("integrity_hash") or ""),
                    "sample_index_start": raw_entry.get("sample_index_start"),
                    "sample_count": raw_entry.get("sample_count"),
                }
                entries.append(((entry["received_ns"], stream_key, journal_epoch, journal_sequence), entry))

    claimed_paths: list[Path] = []
    claimed_stream_keys = set(active_stream_keys)
    for _, entry in sorted(entries, key=lambda item: item[0]):
        stream_key = str(entry["stream_key"])
        if stream_key in claimed_stream_keys:
            continue

        journal_id = str(entry["journal_id"])
        claim_path = processing_dir / f"{journal_id}.journal.json"
        try:
            with claim_path.open("x", encoding="utf-8") as handle:
                json.dump(entry, handle)
                handle.write("\n")
        except FileExistsError:
            claimed_stream_keys.add(stream_key)
            continue

        if _cursor_covers_entry(
            state_store.get_cursor(consumer_name, stream_key),
            int(entry["journal_epoch"]),
            int(entry["journal_sequence"]),
        ):
            with contextlib.suppress(FileNotFoundError):
                claim_path.unlink()
            continue

        claimed_paths.append(claim_path)
        claimed_stream_keys.add(stream_key)

    return claimed_paths


def _processing_stream_keys(processing_dir: Path) -> set[str]:
    if not processing_dir.exists():
        return set()
    stream_keys: set[str] = set()
    for claim_path in processing_dir.glob("*.journal.json"):
        with contextlib.suppress(Exception):
            metadata = json.loads(claim_path.read_text(encoding="utf-8"))
            stream_key = str(metadata.get("stream_key") or "")
            if stream_key:
                stream_keys.add(stream_key)
    for claim_path in processing_dir.glob("*.rustdsp.json"):
        with contextlib.suppress(Exception):
            metadata = json.loads(claim_path.read_text(encoding="utf-8"))
            for stream_key in metadata.get("stream_keys", []):
                stream_key_text = str(stream_key or "")
                if stream_key_text:
                    stream_keys.add(stream_key_text)
    return stream_keys


def _lease_existing_processing_claims_oldest_first(
    processing_dir: Path,
    lease_owner: str,
) -> list[Path]:
    if not processing_dir.exists():
        return []

    candidates: list[Path] = []
    candidates.extend(processing_dir.glob("*.journal.json"))
    candidates.extend(processing_dir.glob("*.rustdsp.json"))

    leased: list[Path] = []
    for claim_path in sorted(candidates, key=lambda path: path.stat().st_mtime_ns):
        lease_path = claim_path.with_name(f"lease-{lease_owner}-{claim_path.name}")
        try:
            claim_path.replace(lease_path)
        except FileNotFoundError:
            continue
        leased.append(lease_path)
    return leased


def _load_claimed_rust_dsp_item(
    claim_path: Path,
    journal_streams_dir: Path,
) -> ClaimedLocalizedRenderItem | ClaimedLocalizedNodeHeartbeatItem:
    with claim_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    claim_kind = str(metadata.get("claim_kind") or "localized_render")
    primary_manifest_path = claim_path.parent / str(metadata["primary_manifest_filename"])
    paired_manifest_filename = metadata.get("paired_manifest_filename")
    paired_manifest_path = claim_path.parent / str(paired_manifest_filename) if paired_manifest_filename else None
    primary_manifest_payload = json.loads(primary_manifest_path.read_text(encoding="utf-8"))
    paired_manifest_payload = (
        json.loads(paired_manifest_path.read_text(encoding="utf-8"))
        if paired_manifest_path is not None and paired_manifest_path.exists()
        else None
    )
    render_body_filename = metadata.get("render_body_filename")
    render_body_path: Path | None = None
    if render_body_filename:
        render_body_path = claim_path.parent / str(render_body_filename)
        if not render_body_path.is_file():
            raise FileNotFoundError(
                f"Localized render body file missing for {claim_path.name}: {render_body_filename}"
            )
        _rewrite_manifest_derived_handle_to_processing_copy(
            primary_manifest_payload,
            render_body_path,
        )
        if paired_manifest_payload is not None:
            _rewrite_manifest_derived_handle_to_processing_copy(
                paired_manifest_payload,
                render_body_path,
            )
    cleanup_paths: list[Path] = [claim_path, primary_manifest_path]
    if paired_manifest_path is not None:
        cleanup_paths.append(paired_manifest_path)
    if render_body_path is not None:
        cleanup_paths.append(render_body_path)

    if claim_kind == "node_heartbeat":
        node_context_payload = primary_manifest_payload.get("node_context")
        if not isinstance(node_context_payload, dict):
            raise ValueError(f"Missing node_context in heartbeat claim {claim_path.name}")
        node_payload = node_context_payload.get("node")
        if not isinstance(node_payload, dict):
            raise ValueError(f"Missing node payload in heartbeat claim {claim_path.name}")
        return ClaimedLocalizedNodeHeartbeatItem(
            ingest_id=str(metadata.get("ingest_id") or primary_manifest_payload.get("manifest_id") or claim_path.stem),
            source_kind="journal_localized_render_manifest",
            received_ns=int(metadata.get("received_ns") or primary_manifest_payload.get("created_ns") or time.time_ns()),
            cleanup_paths=tuple(cleanup_paths),
            cursor_updates=tuple(),
            node=NodeSpec.model_validate(node_payload),
        )

    bundle = load_localized_render_manifest_bundle(
        manifest_payload=primary_manifest_payload,
        journal_streams_dir=journal_streams_dir,
        paired_classifier_manifest_payload=paired_manifest_payload,
    )
    return ClaimedLocalizedRenderItem(
        ingest_id=str(metadata.get("ingest_id") or primary_manifest_payload.get("manifest_id") or claim_path.stem),
        source_kind="journal_localized_render_manifest",
        received_ns=int(metadata.get("received_ns") or primary_manifest_payload.get("created_ns") or time.time_ns()),
        cleanup_paths=tuple(cleanup_paths),
        cursor_updates=bundle.cursor_updates,
        localized_render_request=bundle.request,
    )


def _claim_rust_dsp_manifests_oldest_first(
    journal_manifest_dir: Path,
    journal_streams_dir: Path,
    processing_dir: Path,
    journal_consumer_state_path: Path,
    consumer_name: str,
    max_claims: int | None = None,
) -> list[Path]:
    if not journal_manifest_dir.exists():
        return []

    processing_dir.mkdir(parents=True, exist_ok=True)
    state_store = JournalConsumerStateStore(journal_consumer_state_path)
    state_store.ensure_initialized()
    cursor_by_stream = state_store.load_cursors(consumer_name)

    localization_candidates: dict[tuple[tuple[Any, ...], ...], tuple[Path, dict[str, Any]]] = {}
    classifier_candidates: dict[tuple[tuple[Any, ...], ...], tuple[Path, dict[str, Any]]] = {}
    heartbeat_candidates: dict[tuple[tuple[Any, ...], ...], tuple[Path, dict[str, Any]]] = {}

    manifest_paths = [path for path in journal_manifest_dir.glob("*.json") if path.is_file()]
    if max_claims is not None:
        scan_limit = max(256, int(max_claims) * 64)
        manifest_paths = sorted(
            manifest_paths,
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )[:scan_limit]
    else:
        manifest_paths = sorted(manifest_paths)

    for manifest_path in manifest_paths:
        if not manifest_path.is_file() or manifest_path.name.startswith("."):
            continue
        with contextlib.suppress(Exception):
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_type = str(manifest_payload.get("manifest_type") or "")
            source_key = manifest_source_key(manifest_payload)
            if not source_key:
                continue
            typed_source_key = cast(tuple[tuple[Any, ...], ...], source_key)
            if manifest_type == "localization_result" and manifest_payload.get("classifier_render") is not None:
                localization_candidates[typed_source_key] = (manifest_path, manifest_payload)
            elif (
                manifest_type == "localization_result"
                and manifest_payload.get("classifier_render") is None
                and isinstance(manifest_payload.get("node_context"), dict)
            ):
                heartbeat_candidates[typed_source_key] = (manifest_path, manifest_payload)
            elif manifest_type == "classifier_render" and manifest_payload.get("derived_handle") is not None:
                classifier_candidates[typed_source_key] = (manifest_path, manifest_payload)

    claims: list[tuple[int, str, Path, dict[str, Any], Path | None, dict[str, Any] | None]] = []
    for source_key, (localization_path, localization_payload) in localization_candidates.items():
        classifier_path, classifier_payload = classifier_candidates.get(source_key, (None, None))
        claims.append(
            (
                int(localization_payload.get("created_ns") or 0),
                "localized_render",
                localization_path,
                localization_payload,
                classifier_path,
                classifier_payload,
            )
        )
    for source_key, (classifier_path, classifier_payload) in classifier_candidates.items():
        if source_key in localization_candidates:
            continue
        claims.append(
            (
                int(classifier_payload.get("created_ns") or 0),
                "localized_render",
                classifier_path,
                classifier_payload,
                None,
                None,
            )
        )
    for source_key, (heartbeat_path, heartbeat_payload) in heartbeat_candidates.items():
        if source_key in localization_candidates or source_key in classifier_candidates:
            continue
        claims.append(
            (
                int(heartbeat_payload.get("created_ns") or 0),
                "node_heartbeat",
                heartbeat_path,
                heartbeat_payload,
                None,
                None,
            )
        )

    claimed_paths: list[Path] = []
    claim_limit = max_claims if max_claims is None else max(0, int(max_claims))
    for _, claim_kind, primary_path, primary_payload, paired_path, paired_payload in sorted(
        claims,
        key=lambda item: item[0],
        reverse=max_claims is not None,
    ):
        if claim_limit is not None and len(claimed_paths) >= claim_limit:
            break
        if _manifest_sources_are_cursor_covered(
            primary_payload,
            cursor_by_stream,
            journal_streams_dir,
        ):
            with contextlib.suppress(FileNotFoundError):
                primary_path.unlink()
            if paired_path is not None:
                with contextlib.suppress(FileNotFoundError):
                    paired_path.unlink()
            continue
        with contextlib.suppress(Exception):
            if claim_kind == "node_heartbeat":
                cursor_updates = _cursor_updates_for_manifest_source_handles(
                    primary_payload,
                    journal_streams_dir,
                )
                stream_keys = {cursor_update.stream_key for cursor_update in cursor_updates}
                ingest_id = str(primary_payload.get("manifest_id") or primary_path.stem)
                primary_processing_path = processing_dir / primary_path.name
                primary_path.replace(primary_processing_path)
                claim_payload = {
                    "source_kind": "journal_localized_render_manifest",
                    "claim_kind": "node_heartbeat",
                    "ingest_id": ingest_id,
                    "received_ns": time.time_ns(),
                    "stream_keys": sorted(stream_keys),
                    "cursor_updates": [],
                    "primary_manifest_filename": primary_processing_path.name,
                    "paired_manifest_filename": None,
                    "render_body_filename": None,
                    "cleanup_filenames": [primary_processing_path.name],
                }
                claim_path = processing_dir / f"{ingest_id}.rustdsp.json"
                with claim_path.open("x", encoding="utf-8") as handle:
                    json.dump(claim_payload, handle)
                    handle.write("\n")
                claimed_paths.append(claim_path)
                continue

            bundle = load_localized_render_manifest_bundle(
                manifest_payload=primary_payload,
                journal_streams_dir=journal_streams_dir,
                paired_classifier_manifest_payload=paired_payload,
            )
            stream_keys = {cursor_update.stream_key for cursor_update in bundle.cursor_updates}
            if bundle.cursor_updates and all(
                _cursor_covers_entry(
                    cursor_by_stream.get(cursor_update.stream_key),
                    cursor_update.journal_epoch,
                    cursor_update.journal_sequence,
                )
                for cursor_update in bundle.cursor_updates
            ):
                with contextlib.suppress(FileNotFoundError):
                    primary_path.unlink()
                if paired_path is not None:
                    with contextlib.suppress(FileNotFoundError):
                        paired_path.unlink()
                continue

            primary_processing_path = processing_dir / primary_path.name
            paired_processing_path: Path | None = None
            primary_path.replace(primary_processing_path)
            if paired_path is not None:
                paired_processing_path = processing_dir / paired_path.name
                paired_path.replace(paired_processing_path)
            render_body_filename = _materialize_processing_local_render_body(
                processing_dir=processing_dir,
                ingest_id=bundle.request.manifest_id,
                primary_manifest_payload=primary_payload,
                paired_manifest_payload=paired_payload,
            )

            claim_payload = {
                "source_kind": "journal_localized_render_manifest",
                "claim_kind": "localized_render",
                "ingest_id": bundle.request.manifest_id,
                "received_ns": time.time_ns(),
                "stream_keys": sorted(stream_keys),
                "cursor_updates": [
                    {
                        "journal_id": cursor_update.journal_id,
                        "stream_key": cursor_update.stream_key,
                        "journal_epoch": cursor_update.journal_epoch,
                        "journal_sequence": cursor_update.journal_sequence,
                    }
                    for cursor_update in bundle.cursor_updates
                ],
                "primary_manifest_filename": primary_processing_path.name,
                "paired_manifest_filename": paired_processing_path.name if paired_processing_path is not None else None,
                "render_body_filename": render_body_filename,
                "cleanup_filenames": [
                    name
                    for name in (
                        primary_processing_path.name,
                        paired_processing_path.name if paired_processing_path is not None else None,
                        render_body_filename,
                    )
                    if name is not None
                ],
            }
            claim_path = processing_dir / f"{bundle.request.manifest_id}.rustdsp.json"
            with claim_path.open("x", encoding="utf-8") as handle:
                json.dump(claim_payload, handle)
                handle.write("\n")
            claimed_paths.append(claim_path)

    return claimed_paths


def _manifest_sources_are_cursor_covered(
    manifest_payload: dict[str, Any],
    cursor_by_stream: dict[str, JournalConsumerCursor],
    journal_streams_dir: Path,
) -> bool:
    source_handles = manifest_payload.get("source_handles")
    if not isinstance(source_handles, list) or not source_handles:
        return False
    covered_any = False
    for handle_payload in source_handles:
        if not isinstance(handle_payload, dict):
            return False
        stream_key = str(handle_payload.get("stream_key") or "")
        journal_epoch = _optional_int(handle_payload.get("journal_epoch"))
        if not stream_key or journal_epoch is None:
            return False
        journal_sequence = _journal_sequence_from_journal_id(
            str(handle_payload.get("journal_id") or "")
        )
        if journal_sequence is None:
            journal_sequence = _journal_sequence_from_manifest_source_handle(handle_payload)
        if journal_sequence is None:
            with contextlib.suppress(Exception):
                handle = JournalPayloadHandle.from_mapping(
                    handle_payload,
                    journal_streams_dir=journal_streams_dir,
                )
                cursor_update = _lookup_cursor_update(handle, journal_streams_dir)
                journal_sequence = cursor_update.journal_sequence
        if journal_sequence is None:
            return False
        if not _cursor_covers_entry(cursor_by_stream.get(stream_key), journal_epoch, journal_sequence):
            return False
        covered_any = True
    return covered_any


def _journal_sequence_from_manifest_source_handle(handle_payload: dict[str, Any]) -> int | None:
    for raw_value in (
        handle_payload.get("journal_sequence"),
        handle_payload.get("sequence"),
    ):
        parsed = _optional_int(raw_value)
        if parsed is not None:
            return parsed
    return None


def _journal_sequence_from_journal_id(value: str) -> int | None:
    if not value:
        return None
    parts = value.split("-")
    if len(parts) < 3 or parts[0] != "jrnl":
        return None
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return None


def _materialize_processing_local_render_body(
    *,
    processing_dir: Path,
    ingest_id: str,
    primary_manifest_payload: dict[str, Any],
    paired_manifest_payload: dict[str, Any] | None,
) -> str | None:
    derived_handle_payload = primary_manifest_payload.get("derived_handle")
    if not isinstance(derived_handle_payload, dict) and paired_manifest_payload is not None:
        derived_handle_payload = paired_manifest_payload.get("derived_handle")
    if not isinstance(derived_handle_payload, dict):
        return None

    derived_handle = JournalPayloadHandle.from_mapping(derived_handle_payload)
    render_body_path = processing_dir / f"{ingest_id}.render.body"
    render_body_path.write_bytes(derived_handle.read_bytes())
    return render_body_path.name


def _cursor_updates_for_manifest_source_handles(
    manifest_payload: dict[str, Any],
    journal_streams_dir: Path,
) -> tuple[JournalCursorUpdate, ...]:
    source_handles = manifest_payload.get("source_handles")
    if not isinstance(source_handles, list):
        return tuple()
    updates: list[JournalCursorUpdate] = []
    for source_handle_payload in source_handles:
        if not isinstance(source_handle_payload, dict):
            continue
        with contextlib.suppress(Exception):
            handle = JournalPayloadHandle.from_mapping(
                source_handle_payload,
                journal_streams_dir=journal_streams_dir,
            )
            updates.append(_lookup_cursor_update(handle, journal_streams_dir))
    return tuple(updates)


def _rewrite_manifest_derived_handle_to_processing_copy(
    manifest_payload: dict[str, Any],
    render_body_path: Path,
) -> None:
    derived_handle_payload = manifest_payload.get("derived_handle")
    if not isinstance(derived_handle_payload, dict):
        return
    derived_handle_payload["segment_path"] = str(render_body_path)
    derived_handle_payload["payload_offset_bytes"] = 0
    derived_handle_payload["payload_length_bytes"] = render_body_path.stat().st_size
    derived_handle_payload["body_offset_bytes"] = 0
    derived_handle_payload["body_length_bytes"] = render_body_path.stat().st_size


def _cursor_covers_entry(
    cursor: JournalConsumerCursor | None,
    journal_epoch: int,
    journal_sequence: int,
) -> bool:
    if cursor is None:
        return False
    if cursor.journal_epoch != journal_epoch:
        return cursor.journal_epoch > journal_epoch
    return cursor.last_fully_processed_journal_sequence >= journal_sequence


def _journal_id_from_claim_path(claim_path: Path) -> str:
    return claim_path.name.removesuffix(".journal.json")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
