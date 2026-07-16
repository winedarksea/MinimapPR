"""Project-bundled YAMNet SavedModel asset loading and validation."""

from __future__ import annotations

from pathlib import Path


_YAMNET_ASSET_DIRECTORY = Path(__file__).resolve().parents[1] / "assets" / "yamnet"
_REQUIRED_YAMNET_MODEL_FILES = (
    "saved_model.pb",
    "variables/variables.index",
    "variables/variables.data-00000-of-00001",
    "assets/yamnet_class_map.csv",
)


def yamnet_model_asset_directory() -> Path:
    """Return the version-controlled SavedModel directory shipped with MinimapPR."""
    return _YAMNET_ASSET_DIRECTORY


def validate_yamnet_model_asset(model_directory: Path | None = None) -> Path:
    """Ensure the bundled model has every file required for inference."""
    resolved_model_directory = Path(model_directory or _YAMNET_ASSET_DIRECTORY)
    missing_paths = [
        relative_path
        for relative_path in _REQUIRED_YAMNET_MODEL_FILES
        if not (resolved_model_directory / relative_path).is_file()
    ]
    if missing_paths:
        missing = ", ".join(missing_paths)
        raise RuntimeError(
            "Bundled YAMNet model is incomplete at "
            f"{resolved_model_directory}: missing {missing}. Restore the repository's "
            "minimappr/assets/yamnet asset before starting MinimapPR."
        )
    return resolved_model_directory


def load_bundled_yamnet_model():
    """Load the immutable local SavedModel without consulting TensorFlow Hub."""
    try:
        import tensorflow as tf  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("YAMNet backend requires tensorflow") from exc

    model_directory = validate_yamnet_model_asset()
    try:
        return tf.saved_model.load(str(model_directory))
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load bundled YAMNet model at {model_directory}. "
            "Verify the checked-in asset was not modified."
        ) from exc


__all__ = [
    "load_bundled_yamnet_model",
    "validate_yamnet_model_asset",
    "yamnet_model_asset_directory",
]
