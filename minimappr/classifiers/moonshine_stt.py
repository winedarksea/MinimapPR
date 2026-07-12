"""Direct ONNX Runtime Moonshine speech-to-text transcriber.

The model is fetched from Hugging Face once on first utterance, rather than
depending on the unmaintained ``useful-moonshine-onnx`` wrapper.  Loading is
invoked by :class:`SpeechCaptureManager` in a worker thread, keeping model I/O
and autoregressive decoding off the audio ingest path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

_TARGET_SAMPLE_RATE_HZ = 16_000
_ENCODER_FILENAME = "encoder_model_quantized.onnx"
_DECODER_FILENAME = "decoder_model_merged_quantized.onnx"


class MoonshineUnavailableError(RuntimeError):
    """Raised when the configured direct ONNX Moonshine runtime cannot load."""


class MoonshineTranscriber:
    """Lazy-ready Moonshine base transcriber backed by quantized ONNX models."""

    def __init__(
        self,
        model_id: str = "onnx-community/moonshine-base-ONNX",
        cache_dir: Path | str | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
            from huggingface_hub import snapshot_download
            from transformers import AutoConfig, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MoonshineUnavailableError(
                "Moonshine STT requires the 'stt' extra "
                "(pip install 'minimappr[stt]')"
            ) from exc

        try:
            snapshot_path = snapshot_download(
                repo_id=model_id,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                allow_patterns=[
                    "config.json",
                    "generation_config.json",
                    "tokenizer*",
                    "special_tokens_map.json",
                    "added_tokens.json",
                    "onnx/encoder_model_quantized.onnx",
                    "onnx/decoder_model_merged_quantized.onnx",
                ],
            )
            model_path = Path(snapshot_path)
            encoder_path = model_path / "onnx" / _ENCODER_FILENAME
            decoder_path = model_path / "onnx" / _DECODER_FILENAME
            if not encoder_path.is_file() or not decoder_path.is_file():
                raise FileNotFoundError("Moonshine ONNX encoder or decoder artifact is missing")
            self._config = AutoConfig.from_pretrained(model_path, local_files_only=True)
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self._encoder_session = ort.InferenceSession(str(encoder_path))
            self._decoder_session = ort.InferenceSession(str(decoder_path))
        except Exception as exc:  # noqa: BLE001 - model download/load boundary
            raise MoonshineUnavailableError(
                f"Unable to load Moonshine ONNX model {model_id!r}: {exc}"
            ) from exc

        self._eos_token_id = int(self._config.eos_token_id)
        self._decoder_start_token_id = int(self._config.decoder_start_token_id)
        self._decoder_layers = int(self._config.decoder_num_hidden_layers)
        self._key_value_heads = int(self._config.decoder_num_key_value_heads)
        self._key_value_dimension = int(
            self._config.hidden_size // self._config.decoder_num_attention_heads
        )
        self._max_positions = int(self._config.max_position_embeddings)

    def transcribe(self, samples: np.ndarray, sample_rate_hz: int) -> str:
        waveform = samples.astype(np.float32, copy=False)
        if sample_rate_hz != _TARGET_SAMPLE_RATE_HZ and waveform.size > 0:
            waveform = resample_poly(
                waveform, up=_TARGET_SAMPLE_RATE_HZ, down=sample_rate_hz
            ).astype(np.float32)
        if waveform.size == 0:
            return ""

        audio = np.ascontiguousarray(waveform[None, :], dtype=np.float32)
        encoder_outputs = self._encoder_session.run(None, {"input_values": audio})[0]
        generated_tokens = np.array([[self._decoder_start_token_id]], dtype=np.int64)
        max_tokens = min(
            max(1, int((audio.shape[-1] / _TARGET_SAMPLE_RATE_HZ) * 6)),
            self._max_positions,
        )
        past_key_values = self._empty_past_key_values(batch_size=audio.shape[0])

        for token_index in range(max_tokens):
            use_cache_branch = token_index > 0
            decoder_inputs: dict[str, Any] = {
                "input_ids": generated_tokens[:, -1:],
                "encoder_hidden_states": encoder_outputs,
                "use_cache_branch": np.asarray([use_cache_branch], dtype=np.bool_),
                **past_key_values,
            }
            outputs = self._decoder_session.run(None, decoder_inputs)
            logits = outputs[0]
            next_tokens = logits[:, -1, :].argmax(axis=-1, keepdims=True).astype(np.int64)
            self._update_past_key_values(
                past_key_values, outputs[1:], use_cache_branch=use_cache_branch
            )
            generated_tokens = np.concatenate((generated_tokens, next_tokens), axis=-1)
            if np.all(next_tokens == self._eos_token_id):
                break

        decoded = self._tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return str(decoded[0]).strip() if decoded else ""

    def _empty_past_key_values(self, *, batch_size: int) -> dict[str, np.ndarray]:
        return {
            f"past_key_values.{layer}.{module}.{key_or_value}": np.zeros(
                (batch_size, self._key_value_heads, 0, self._key_value_dimension),
                dtype=np.float32,
            )
            for layer in range(self._decoder_layers)
            for module in ("decoder", "encoder")
            for key_or_value in ("key", "value")
        }

    def _update_past_key_values(
        self,
        past_key_values: dict[str, np.ndarray],
        present_values: list[np.ndarray],
        *,
        use_cache_branch: bool,
    ) -> None:
        # Merged Moonshine decoder cache outputs follow the cache-input order.
        # Encoder cache is immutable after the first pass.
        for cache_name, present_value in zip(past_key_values, present_values):
            if not use_cache_branch or ".decoder." in cache_name:
                past_key_values[cache_name] = present_value


__all__ = ["MoonshineTranscriber", "MoonshineUnavailableError"]
