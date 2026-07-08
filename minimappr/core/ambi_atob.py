"""Compatibility exports for the spatial-audio A-to-B encoder.

The implementation lives in :mod:`minimappr.spatial_audio.linear_atob`; this
module keeps legacy imports stable for the rest of the codebase.
"""

from minimappr.spatial_audio.linear_atob import (
    ALIAS_CUTOFF_HZ,
    SPEED_OF_SOUND_MPS,
    SIRITH_MIC_POSITIONS_M,
    TIKHONOV_F_REF_HZ,
    TIKHONOV_LAMBDA_0,
    _build_encoding_matrix,
    alias_cutoff_from_positions,
    atob_foa,
    centroid_corrected_positions,
    encode_mono_to_bformat,
    foa_geometry_suitable,
)

__all__ = [
    "ALIAS_CUTOFF_HZ",
    "SPEED_OF_SOUND_MPS",
    "SIRITH_MIC_POSITIONS_M",
    "TIKHONOV_F_REF_HZ",
    "TIKHONOV_LAMBDA_0",
    "_build_encoding_matrix",
    "alias_cutoff_from_positions",
    "atob_foa",
    "centroid_corrected_positions",
    "encode_mono_to_bformat",
    "foa_geometry_suitable",
]
