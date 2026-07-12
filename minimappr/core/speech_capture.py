"""Speech utterance capture: pre-roll + continuation buffering, transcription, persistence.

Owned by :class:`~minimappr.core.fusion_node.FusionNode`. One capture session per
node — triggered when a routing-config trigger sees a YAMNet "Speech" score above
threshold, extended by subsequent speech hits, closed by a watchdog on hangover or
a hard utterance cap.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np

from minimappr.classifiers.moonshine_stt import MoonshineTranscriber, MoonshineUnavailableError
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.models import TranscriptRecord
from minimappr.utils.audio import write_wav_mono

logger = logging.getLogger(__name__)

TranscriptConsumer = Callable[[TranscriptRecord], Awaitable[None]]

_WATCHDOG_INTERVAL_SECONDS = 0.1


@dataclass
class _Session:
    node_id: str
    sensor_id: str
    start_ns: int
    last_speech_ns: int
    trigger_confidence: float
    watchdog_task: "asyncio.Task[None] | None" = field(default=None, repr=False)
    closing: bool = False


class SpeechCaptureManager:
    def __init__(
        self,
        *,
        settings: Any,
        buffer: MultiSensorBuffer,
        storage: Any,
        transcriber_factory: Callable[[], MoonshineTranscriber] | None = None,
    ) -> None:
        self._settings = settings
        self._buffer = buffer
        self._storage = storage
        self._transcriber_factory = transcriber_factory or (
            lambda: MoonshineTranscriber(
                model_id=self._settings.stt_model_id,
                cache_dir=self._settings.stt_model_cache_dir,
            )
        )
        self._transcriber: MoonshineTranscriber | None = None
        self._transcriber_unavailable = False
        self._sessions: dict[str, _Session] = {}
        self._consumers: list[TranscriptConsumer] = []
        self._transcribe_semaphore = asyncio.Semaphore(1)

    def add_consumer(self, consumer: TranscriptConsumer) -> None:
        self._consumers.append(consumer)

    def close(self) -> None:
        for session in list(self._sessions.values()):
            if session.watchdog_task is not None:
                session.watchdog_task.cancel()
            self._buffer.unpin(self._session_id(session.node_id))

    async def maybe_trigger(
        self,
        *,
        node_id: str,
        sensor_id: str,
        event_time_ns: int,
        speech_confidence: float,
    ) -> None:
        if not self._settings.stt_enabled:
            return
        if speech_confidence < self._settings.stt_trigger_min_confidence:
            return
        if self._transcriber_unavailable:
            return

        existing = self._sessions.get(node_id)
        if existing is not None:
            existing.last_speech_ns = max(existing.last_speech_ns, event_time_ns)
            return

        pre_roll_ns = int(self._settings.stt_pre_roll_seconds * 1_000_000_000)
        start_ns = max(0, event_time_ns - pre_roll_ns)
        session = _Session(
            node_id=node_id,
            sensor_id=sensor_id,
            start_ns=start_ns,
            last_speech_ns=event_time_ns,
            trigger_confidence=speech_confidence,
        )
        self._sessions[node_id] = session
        self._buffer.pin(self._session_id(node_id), start_ns)
        session.watchdog_task = asyncio.create_task(
            self._watchdog(session), name=f"speech-capture-watchdog-{node_id}"
        )

    def _session_id(self, node_id: str) -> str:
        return f"speech-capture-{node_id}"

    async def _watchdog(self, session: _Session) -> None:
        hangover_ns = int(self._settings.stt_hangover_seconds * 1_000_000_000)
        max_span_ns = int(self._settings.stt_max_utterance_seconds * 1_000_000_000)
        try:
            while True:
                await asyncio.sleep(_WATCHDOG_INTERVAL_SECONDS)
                now_ns = time.time_ns()
                idle_ns = now_ns - session.last_speech_ns
                span_ns = now_ns - session.start_ns
                if idle_ns >= hangover_ns or span_ns >= max_span_ns:
                    await self._close_session(session)
                    return
        except asyncio.CancelledError:
            raise

    async def _close_session(self, session: _Session) -> None:
        if session.closing:
            return
        session.closing = True
        self._sessions.pop(session.node_id, None)
        session_id = self._session_id(session.node_id)
        end_ns = min(time.time_ns(), session.last_speech_ns + int(
            self._settings.stt_hangover_seconds * 1_000_000_000
        ))
        try:
            audio, sample_rate_hz, _diag = await self._buffer.extract_range(
                [session.sensor_id], session.start_ns, end_ns
            )
        finally:
            self._buffer.unpin(session_id)

        if audio.size == 0:
            return

        async with self._transcribe_semaphore:
            text = await asyncio.to_thread(self._transcribe, audio, sample_rate_hz)
        if not text:
            return

        transcript_id = f"txt-{uuid.uuid4().hex[:16]}"
        audio_dir = Path(self._settings.speech_audio_dir)
        audio_path = audio_dir / f"{transcript_id}.wav"
        await asyncio.to_thread(write_wav_mono, audio_path, audio, sample_rate_hz)

        created_ns = time.time_ns()
        await self._storage.insert_transcript(
            transcript_id=transcript_id,
            node_id=session.node_id,
            sensor_id=session.sensor_id,
            start_ns=session.start_ns,
            end_ns=end_ns,
            text=text,
            model="moonshine",
            trigger_confidence=session.trigger_confidence,
            audio_path=str(audio_path),
            detection_id=None,
            created_ns=created_ns,
        )
        record = TranscriptRecord(
            id=transcript_id,
            node_id=session.node_id,
            sensor_id=session.sensor_id,
            start_ns=session.start_ns,
            end_ns=end_ns,
            text=text,
            model="moonshine",
            trigger_confidence=session.trigger_confidence,
            audio_path=str(audio_path),
            detection_id=None,
            created_ns=created_ns,
        )
        for consumer in self._consumers:
            try:
                await consumer(record)
            except Exception:  # noqa: BLE001 — one bad consumer must not drop the transcript
                logger.exception("Transcript consumer failed for %s", transcript_id)

    def _transcribe(self, audio: np.ndarray, sample_rate_hz: int) -> str:
        if self._transcriber is None:
            try:
                self._transcriber = self._transcriber_factory()
            except MoonshineUnavailableError as exc:
                self._transcriber_unavailable = True
                logger.warning("Speech capture disabled: %s", exc)
                return ""
        try:
            return self._transcriber.transcribe(audio, sample_rate_hz)
        except Exception:  # noqa: BLE001
            logger.exception("Moonshine transcription failed")
            return ""


__all__ = ["SpeechCaptureManager", "TranscriptConsumer"]
