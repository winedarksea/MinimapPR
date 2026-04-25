"""Chunked dispatch policy for long-window classifiers like BirdNET."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ClassificationChunkingPolicy:
    """Allow at most one classification dispatch per source node per chunk.

    Long-window classifiers do not benefit from evaluating heavily overlapping
    windows for every trigger candidate. A chunked policy keeps overlap
    intentional and bounded so the classifier can stay near real time.
    """

    stride_seconds: float
    _stride_ns: int = field(init=False)
    _last_chunk_id_by_node: dict[str, int] = field(init=False, default_factory=dict)
    _retry_allowed_by_node: dict[str, bool] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._stride_ns = max(1, int(self.stride_seconds * 1_000_000_000))

    def should_dispatch(self, *, source_node_id: str, event_time_ns: int) -> bool:
        chunk_id = int(event_time_ns) // self._stride_ns
        previous_chunk_id = self._last_chunk_id_by_node.get(source_node_id)
        if previous_chunk_id == chunk_id:
            retry_allowed = bool(self._retry_allowed_by_node.get(source_node_id, False))
            if not retry_allowed:
                return False
            # Consume a single retry token for this chunk. If this retry is
            # still non-actionable, record_dispatch_outcome(...) will re-open it.
            self._retry_allowed_by_node[source_node_id] = False
            return True
        self._last_chunk_id_by_node[source_node_id] = chunk_id
        self._retry_allowed_by_node[source_node_id] = False
        return True

    def record_dispatch_outcome(
        self,
        *,
        source_node_id: str,
        event_time_ns: int,
        produced_actionable_detection: bool,
    ) -> None:
        """Record whether a chunk dispatch produced an actionable detection.

        Non-actionable outputs (for example, unknown/0.0 confidence) allow a
        retry within the same chunk so early low-context dispatches do not
        suppress later, richer context in that chunk.
        """
        chunk_id = int(event_time_ns) // self._stride_ns
        self._last_chunk_id_by_node[source_node_id] = chunk_id
        self._retry_allowed_by_node[source_node_id] = not produced_actionable_detection
