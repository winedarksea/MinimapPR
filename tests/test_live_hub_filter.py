from __future__ import annotations

from minimappr.api.live import LiveEventHub, LiveSubscriptionFilter


def test_node_state_events_bypass_content_filters() -> None:
    """A subscriber with content filters must still receive node-state events.

    node_updated / node_capability_status payloads carry no detection or track, so
    the content filters (categories/labels/zones/statuses) do not apply to them.
    """
    flt = LiveSubscriptionFilter(label_categories={"vehicle"}, labels={"truck"})

    node_updated = {"type": "node_updated", "node": {"id": "node-1"}}
    node_capability = {
        "type": "node_capability_status",
        "node_id": "node-1",
        "capability": "ptz_camera",
        "status": {"state": "idle"},
    }

    assert LiveEventHub._matches_filter(node_updated, flt) is True
    assert LiveEventHub._matches_filter(node_capability, flt) is True


def test_detection_events_still_filtered() -> None:
    """Content filters still gate detection/track payloads."""
    flt = LiveSubscriptionFilter(label_categories={"vehicle"})

    matching = {"detection": {"label_category": "vehicle"}}
    non_matching = {"detection": {"label_category": "animal"}}

    assert LiveEventHub._matches_filter(matching, flt) is True
    assert LiveEventHub._matches_filter(non_matching, flt) is False


def test_no_filter_delivers_everything() -> None:
    flt = LiveSubscriptionFilter()
    assert LiveEventHub._matches_filter({"detection": {"label_category": "x"}}, flt) is True
    assert LiveEventHub._matches_filter({"type": "node_updated"}, flt) is True
