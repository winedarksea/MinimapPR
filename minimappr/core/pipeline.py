"""Compatibility shim for older imports.

`ProcessingPipeline` now maps to the queue-driven `FusionNode` runtime.
"""

from __future__ import annotations

from minimappr.core.fusion_node import FusionNode


class ProcessingPipeline(FusionNode):
    """Backwards-compatible alias for the fusion runtime."""

    pass
