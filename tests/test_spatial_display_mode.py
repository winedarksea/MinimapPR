from minimappr.models import spatial_display_mode_for_detection


def test_spatial_display_mode_classifies_node_only_bearing_and_localized() -> None:
    assert spatial_display_mode_for_detection(
        reporting_modality="omni",
        feature_summary={},
    ) == "node_only"

    assert spatial_display_mode_for_detection(
        reporting_modality="localized",
        feature_summary={"localization_range_projection_mode": "range_asymptotic"},
    ) == "bearing_only"

    assert spatial_display_mode_for_detection(
        reporting_modality="localized",
        feature_summary={"localization_range_observability": 0.10},
    ) == "bearing_only"

    assert spatial_display_mode_for_detection(
        reporting_modality="localized",
        feature_summary={"localization_range_observability": 0.42},
    ) == "localized"
