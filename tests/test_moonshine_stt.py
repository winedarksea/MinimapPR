"""Direct ONNX Moonshine decoding tests without model downloads."""

from __future__ import annotations

import numpy as np

from minimappr.classifiers.moonshine_stt import MoonshineTranscriber


class _EncoderSession:
    def run(self, _outputs, inputs):
        assert inputs["input_values"].shape == (1, 16_000)
        return [np.ones((1, 2, 4), dtype=np.float32)]


class _Output:
    def __init__(self, name: str) -> None:
        self.name = name


class _DecoderSession:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get_outputs(self):
        return [_Output("logits"), _Output("present.0.decoder.key")]

    def run(self, _outputs, inputs):
        self.calls.append(inputs)
        logits = np.full((1, 1, 5), -100.0, dtype=np.float32)
        logits[0, 0, 3 if len(self.calls) == 1 else 2] = 1.0
        return [logits, np.ones((1, 1, len(self.calls), 1), dtype=np.float32)]


class _Tokenizer:
    def batch_decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens is True
        assert tokens.tolist() == [[1, 3, 2]]
        return ["  drone overhead  "]


def test_direct_onnx_transcriber_resamples_decodes_and_stops_at_eos() -> None:
    transcriber = MoonshineTranscriber.__new__(MoonshineTranscriber)
    transcriber._encoder_session = _EncoderSession()
    decoder = _DecoderSession()
    transcriber._decoder_session = decoder
    transcriber._tokenizer = _Tokenizer()
    transcriber._eos_token_id = 2
    transcriber._decoder_start_token_id = 1
    transcriber._decoder_layers = 1
    transcriber._key_value_heads = 1
    transcriber._key_value_dimension = 1
    transcriber._max_positions = 32

    text = transcriber.transcribe(np.ones(8_000, dtype=np.float32), sample_rate_hz=8_000)

    assert text == "drone overhead"
    assert len(decoder.calls) == 2
    assert bool(decoder.calls[0]["use_cache_branch"].item()) is False
    assert bool(decoder.calls[1]["use_cache_branch"].item()) is True
    assert decoder.calls[1]["past_key_values.0.decoder.key"].shape[-2] == 1
