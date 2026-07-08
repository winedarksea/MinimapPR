"""Spatial-audio encoders for MinimapPR."""

from minimappr.spatial_audio.geometry import (
    ALIAS_CUTOFF_HZ,
    SIRITH_MIC_POSITIONS_M,
    SPEED_OF_SOUND_MPS,
    NodeOrientation,
    alias_cutoff_from_positions,
    centroid_corrected_positions,
    foa_geometry_suitable,
    rotate_positions,
)
from minimappr.spatial_audio.linear_atob import atob_foa, encode_mono_to_bformat
from minimappr.spatial_audio.parametric import encode_ambisonics, enhance_foa_parametric
from minimappr.spatial_audio.objects import (
    DEFAULT_OBJECT_SUBTRACTION_PROFILE,
    ObjectSubtractionProfile,
    subtract_object_slot_from_bed,
    subtract_objects_from_bed,
)
from minimappr.spatial_audio.profiles import (
    LINEAR_V1,
    PARAMETRIC_V2,
    PROFILES,
    AmbisonicsProfile,
    get_profile,
)

__all__ = [
    "ALIAS_CUTOFF_HZ",
    "SIRITH_MIC_POSITIONS_M",
    "SPEED_OF_SOUND_MPS",
    "AmbisonicsProfile",
    "DEFAULT_OBJECT_SUBTRACTION_PROFILE",
    "LINEAR_V1",
    "NodeOrientation",
    "ObjectSubtractionProfile",
    "PARAMETRIC_V2",
    "PROFILES",
    "alias_cutoff_from_positions",
    "atob_foa",
    "centroid_corrected_positions",
    "encode_ambisonics",
    "encode_mono_to_bformat",
    "enhance_foa_parametric",
    "foa_geometry_suitable",
    "get_profile",
    "rotate_positions",
    "subtract_object_slot_from_bed",
    "subtract_objects_from_bed",
]
