"""SpeechCaptureManager unit tests with a fake transcriber + stub buffer/storage."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from minimappr.core.speech_capture import SpeechCaptureManager
from minimappr.models import TranscriptRecord


class _FakeTranscriber:
    def __init__(self, text: str = "help me please") -> None:
        self.calls = 0
        self._text = text

    def transcribe(self, samples, sample_rate_hz):
        self.calls += 1
        return self._text


class _StubBuffer:
    def __init__(self) -> None:
        self.pins: dict[str, int] = {}
        self.unpin_calls = 0

    def pin(self, session_id, start_ns):
        self.pins[session_id] = start_ns

    def unpin(self, session_id):
        self.unpin_calls += 1
        self.pins.pop(session_id, None)

    async def extract_range(self, sensor_ids, start_ns, end_ns):
        n = max(1, (end_ns - start_ns) // 1_000_000)
        return np.ones(int(n), dtype=np.float32) * 0.1, 16000, {}


class _StubStorage:
    def __init__(self) -> None:
        self.inserted: list[dict] = []

    async def insert_transcript(self, **kwargs):
        self.inserted.append(kwargs)
        return kwargs["transcript_id"]


class _Settings:
    stt_enabled = True
    stt_trigger_min_confidence = 0.5
    stt_pre_roll_seconds = 0.1
    stt_hangover_seconds = 0.2
    stt_max_utterance_seconds = 2.0
    speech_audio_dir = None  # patched per-test


@pytest.mark.asyncio
async def test_trigger_then_hangover_produces_transcript(tmp_path):
    settings = _Settings()
    settings.speech_audio_dir = tmp_path
    buffer = _StubBuffer()
    storage = _StubStorage()
    transcriber = _FakeTranscriber()
    manager = SpeechCaptureManager(
        settings=settings,
        buffer=buffer,
        storage=storage,
        transcriber_factory=lambda: transcriber,
    )
    consumed: list[TranscriptRecord] = []
    manager.add_consumer(lambda record: _capture(consumed, record))

    await manager.maybe_trigger(
        node_id="node-a", sensor_id="node-a:ch0", event_time_ns=1_000_000_000, speech_confidence=0.9
    )
    assert "speech-capture-node-a" in buffer.pins

    await asyncio.sleep(0.5)

    assert len(consumed) == 1
    assert consumed[0].text == "help me please"
    assert transcriber.calls == 1
    assert len(storage.inserted) == 1
    assert buffer.unpin_calls == 1
    assert "speech-capture-node-a" not in buffer.pins
    manager.close()


async def _capture(sink, record):
    sink.append(record)


@pytest.mark.asyncio
async def test_below_threshold_confidence_does_not_trigger(tmp_path):
    settings = _Settings()
    settings.speech_audio_dir = tmp_path
    buffer = _StubBuffer()
    storage = _StubStorage()
    manager = SpeechCaptureManager(
        settings=settings, buffer=buffer, storage=storage, transcriber_factory=_FakeTranscriber
    )
    await manager.maybe_trigger(
        node_id="node-a", sensor_id="node-a:ch0", event_time_ns=1_000_000_000, speech_confidence=0.1
    )
    assert buffer.pins == {}
    manager.close()


@pytest.mark.asyncio
async def test_second_trigger_extends_existing_session_not_new_pin(tmp_path):
    settings = _Settings()
    settings.speech_audio_dir = tmp_path
    buffer = _StubBuffer()
    storage = _StubStorage()
    manager = SpeechCaptureManager(
        settings=settings, buffer=buffer, storage=storage, transcriber_factory=_FakeTranscriber
    )
    await manager.maybe_trigger(
        node_id="node-a", sensor_id="node-a:ch0", event_time_ns=1_000_000_000, speech_confidence=0.9
    )
    first_pin = dict(buffer.pins)
    await manager.maybe_trigger(
        node_id="node-a", sensor_id="node-a:ch0", event_time_ns=1_050_000_000, speech_confidence=0.9
    )
    assert buffer.pins == first_pin  # unchanged: same session extended, not re-pinned
    manager.close()
