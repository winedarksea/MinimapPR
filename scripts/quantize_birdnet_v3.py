#!/usr/bin/env python3
"""INT8-quantize the pruned BirdNET+ V3.0 ONNX model with ONNX Runtime.

Steps:
1. Upcast the mixed-precision FP16 export to FP32 (FP16 initializers -> FP32,
   ``Cast(to=FLOAT16)`` -> ``Cast(to=FLOAT)``). ORT's quantizer calibrates and
   quantizes FP32 graphs only.
2. Exclude the spectrogram front end from quantization: V3 computes its STFT
   as MatMuls against DFT matrices (``preprocess.dft*``), and 8-bit DFT
   weights would wreck the log-mel input. Everything upstream of the first
   ``Conv``'s activation input is left in FP32.
3. Write the requested variants:
   * ``dynamic``: weights INT8, activations quantized at run time
     (MatMul/Gemm + Conv -> ConvInteger). No calibration audio needed.
   * ``static``: QDQ, per-channel INT8 weights, activation ranges calibrated
     on ``--calibration-audio`` files (wav/flac/mp3, any rate).

Calibration audio is untrusted input data: it is only decoded, never run.

Result (2026-11-22, docs/field_mode_progress/birdnet_v3.md): every variant
that quantizes the Conv stack (dynamic, static, with or without SE/depthwise
exclusions) fails the accuracy gate badly; head-only (``--op-types MatMul
Gemm``) matches FP16 but gives no speedup. Treat INT8 as rejected until a
per-layer sensitivity search or QAT is tried.

Usage::

    python scripts/quantize_birdnet_v3.py --variants dynamic static \\
        --calibration-audio tests/house_finch.wav ~/clips/*.wav
    python scripts/quantize_birdnet_v3.py --variants dynamic --op-types MatMul Gemm --output-suffix _head
    python scripts/quantize_birdnet_v3.py --variants static --exclude-weight-patterns .se. conv_dw \\
        --calibration-method minmax --output-suffix _noSEdw --calibration-audio ~/clips
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from minimappr.classifiers.birdnet_v3_audio import (  # noqa: E402
    BirdNETV3WindowPolicy,
    prepare_birdnet_v3_windows,
)

logger = logging.getLogger("quantize_birdnet_v3")

MODEL_DIRECTORY = REPOSITORY_ROOT / "data/models/birdnet_v3"
DEFAULT_SOURCE_MODEL = MODEL_DIRECTORY / "BirdNET+_V3.0_Global_12K_FP16_pruned.onnx"
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg")
ONNX_FLOAT = 1
ONNX_FLOAT16 = 10
# Windows per calibration file; caps runtime when long recordings are given.
DEFAULT_MAX_WINDOWS_PER_FILE = 8
# Calibration windows use one fixed length so every batch has the same shape.
CALIBRATION_WINDOW_SECONDS = 6.0
CALIBRATION_PERCENTILE = 99.99
QUANTIZABLE_OP_TYPES = ("Conv", "MatMul", "Gemm")


def upcast_mixed_precision_to_fp32(source_path: Path, output_path: Path) -> None:
    import onnx  # noqa: PLC0415
    from onnx import numpy_helper  # noqa: PLC0415

    model = onnx.load(str(source_path))
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.data_type == ONNX_FLOAT16:
            upcast = numpy_helper.from_array(numpy_helper.to_array(initializer).astype(np.float32), initializer.name)
            model.graph.initializer[index].CopyFrom(upcast)
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attribute in node.attribute:
                if attribute.name == "to" and attribute.i == ONNX_FLOAT16:
                    attribute.i = ONNX_FLOAT
    for value_info in list(model.graph.value_info):
        if value_info.type.tensor_type.elem_type == ONNX_FLOAT16:
            value_info.type.tensor_type.elem_type = ONNX_FLOAT
    _remove_identity_casts(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))


def _remove_identity_casts(model) -> None:
    """Drop Casts whose input already has the target type (FP32 -> FP32 after upcast).

    The quantizer needs Conv/MatMul weights to be initializers, not Cast outputs.
    """
    import onnx  # noqa: PLC0415

    inferred = onnx.shape_inference.infer_shapes(model)
    element_type_by_name = {
        value.name: value.type.tensor_type.elem_type
        for value in [*inferred.graph.value_info, *inferred.graph.input, *inferred.graph.output]
    }
    element_type_by_name.update({initializer.name: initializer.data_type for initializer in model.graph.initializer})
    graph_outputs = {output.name for output in model.graph.output}
    replacement_by_name: dict[str, str] = {}
    kept_nodes = []
    for node in model.graph.node:
        target_type = next((a.i for a in node.attribute if a.name == "to"), None) if node.op_type == "Cast" else None
        if (
            target_type is not None
            and element_type_by_name.get(node.input[0]) == target_type
            and node.output[0] not in graph_outputs
        ):
            replacement_by_name[node.output[0]] = replacement_by_name.get(node.input[0], node.input[0])
            continue
        kept_nodes.append(node)
    for node in kept_nodes:
        for position, name in enumerate(node.input):
            if name in replacement_by_name:
                node.input[position] = replacement_by_name[name]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    logger.info("removed %d identity Casts", len(replacement_by_name))


def front_end_node_names(model_path: Path) -> list[str]:
    """Names of nodes feeding the first Conv's activation input (the STFT/mel front end)."""
    import onnx  # noqa: PLC0415

    graph = onnx.load(str(model_path)).graph
    producer_by_output = {output: node for node in graph.node for output in node.output}
    first_conv = next(node for node in graph.node if node.op_type == "Conv")
    pending = [first_conv.input[0]]
    front_end: dict[str, None] = {}
    while pending:
        producer = producer_by_output.get(pending.pop())
        if producer is None or producer.name in front_end:
            continue
        front_end[producer.name] = None
        pending.extend(producer.input)
    unnamed = [node for node in graph.node if not node.name]
    if unnamed:
        raise RuntimeError(f"{len(unnamed)} unnamed nodes; cannot build an exclusion list")
    return list(front_end)


def collect_audio_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES))
        elif path.suffix.lower() in AUDIO_SUFFIXES and path.is_file():
            files.append(path)
    return files


def load_mono_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile  # noqa: PLC0415

    audio, sample_rate_hz = soundfile.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), int(sample_rate_hz)


def calibration_windows(audio_files: list[Path], max_windows_per_file: int) -> Iterator[np.ndarray]:
    policy = BirdNETV3WindowPolicy(
        min_window_seconds=CALIBRATION_WINDOW_SECONDS,
        max_window_seconds=CALIBRATION_WINDOW_SECONDS,
        window_overlap_seconds=0.0,
    )
    for path in audio_files:
        try:
            audio, sample_rate_hz = load_mono_audio(path)
        except Exception as error:  # noqa: BLE001 - skip undecodable files, keep calibrating
            logger.warning("skip %s: %s", path, error)
            continue
        windows, verdict = prepare_birdnet_v3_windows(audio, sample_rate_hz, policy)
        if not verdict.accepted:
            continue
        for _, _, window in windows[:max_windows_per_file]:
            yield window


class AudioCalibrationReader:
    """``onnxruntime.quantization.CalibrationDataReader`` over conditioned windows."""

    def __init__(self, input_name: str, windows: list[np.ndarray]) -> None:
        self._feeds = iter([{input_name: window[np.newaxis, :]} for window in windows])

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._feeds, None)


def nodes_matching_weight_patterns(model_path: Path, patterns: list[str]) -> list[str]:
    """Quantizable nodes whose weight input name contains any pattern (e.g. ``.se.``, ``conv_dw``)."""
    import onnx  # noqa: PLC0415

    if not patterns:
        return []
    return [
        node.name
        for node in onnx.load(str(model_path)).graph.node
        if node.op_type in QUANTIZABLE_OP_TYPES
        and len(node.input) > 1
        and any(pattern in node.input[1] for pattern in patterns)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIRECTORY)
    parser.add_argument("--variants", nargs="+", choices=("dynamic", "static"), default=["dynamic", "static"])
    parser.add_argument("--calibration-audio", type=Path, nargs="*", default=[])
    parser.add_argument("--max-windows-per-file", type=int, default=DEFAULT_MAX_WINDOWS_PER_FILE)
    parser.add_argument("--op-types", nargs="+", default=list(QUANTIZABLE_OP_TYPES), choices=QUANTIZABLE_OP_TYPES)
    parser.add_argument(
        "--exclude-weight-patterns",
        nargs="*",
        default=[],
        help="keep nodes in FP32 when their weight name contains one of these (e.g. .se. conv_dw conv_stem head.)",
    )
    parser.add_argument("--calibration-method", choices=("percentile", "minmax", "entropy"), default="percentile")
    parser.add_argument("--output-suffix", default="", help="appended to output names to keep experiment variants apart")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    from onnxruntime.quantization import (  # noqa: PLC0415
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process  # noqa: PLC0415

    stem = args.source_model.stem.replace("_FP16", "")
    fp32_path = args.output_dir / f"{stem}_FP32_upcast.onnx"
    prepared_path = args.output_dir / f"{stem}_FP32_prepared.onnx"
    if not prepared_path.is_file():
        upcast_mixed_precision_to_fp32(args.source_model, fp32_path)
        quant_pre_process(str(fp32_path), str(prepared_path), skip_symbolic_shape=True)
    excluded = front_end_node_names(prepared_path)
    pattern_excluded = nodes_matching_weight_patterns(prepared_path, args.exclude_weight_patterns)
    logger.info("%d front-end + %d pattern-matched nodes kept in FP32", len(excluded), len(pattern_excluded))
    excluded = list(dict.fromkeys(excluded + pattern_excluded))

    if "dynamic" in args.variants:
        dynamic_path = args.output_dir / f"{stem}_INT8_dynamic{args.output_suffix}.onnx"
        quantize_dynamic(
            str(prepared_path),
            str(dynamic_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=args.op_types,
            nodes_to_exclude=excluded,
            per_channel=True,
        )
        logger.info("wrote %s (%.1f MB)", dynamic_path, dynamic_path.stat().st_size / 1e6)

    if "static" in args.variants:
        audio_files = collect_audio_files(args.calibration_audio)
        windows = list(calibration_windows(audio_files, args.max_windows_per_file))
        if not windows:
            logger.error("static quantization needs --calibration-audio with decodable, non-silent clips")
            return 1
        logger.info("calibrating on %d windows from %d files", len(windows), len(audio_files))
        import onnxruntime as ort  # noqa: PLC0415

        input_name = ort.InferenceSession(str(prepared_path), providers=["CPUExecutionProvider"]).get_inputs()[0].name
        static_path = args.output_dir / f"{stem}_INT8_static{args.output_suffix}.onnx"
        method = {
            "percentile": CalibrationMethod.Percentile,
            "minmax": CalibrationMethod.MinMax,
            "entropy": CalibrationMethod.Entropy,
        }[args.calibration_method]
        quantize_static(
            str(prepared_path),
            str(static_path),
            AudioCalibrationReader(input_name, windows),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=args.op_types,
            nodes_to_exclude=excluded,
            calibrate_method=method,
            extra_options={"CalibPercentile": CALIBRATION_PERCENTILE},
        )
        logger.info("wrote %s (%.1f MB)", static_path, static_path.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
