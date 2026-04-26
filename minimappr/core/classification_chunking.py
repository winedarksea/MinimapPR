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
    max_retries_per_chunk: int = 1
    min_retry_progress_seconds: float = 8.0
    _stride_ns: int = field(init=False)
    _min_retry_progress_ns: int = field(init=False)
    _last_chunk_id_by_node: dict[str, int] = field(init=False, default_factory=dict)
    _retry_allowed_by_node: dict[str, bool] = field(init=False, default_factory=dict)
    _retry_count_by_node: dict[str, int] = field(init=False, default_factory=dict)
    _last_dispatch_event_time_ns_by_node: dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._stride_ns = max(1, int(self.stride_seconds * 1_000_000_000))
        self.max_retries_per_chunk = max(0, int(self.max_retries_per_chunk))
        self._min_retry_progress_ns = max(0, int(self.min_retry_progress_seconds * 1_000_000_000))

    def should_dispatch(self, *, source_node_id: str, event_time_ns: int) -> bool:
        chunk_id = int(event_time_ns) // self._stride_ns
        previous_chunk_id = self._last_chunk_id_by_node.get(source_node_id)
        if previous_chunk_id == chunk_id:
            retry_allowed = bool(self._retry_allowed_by_node.get(source_node_id, False))
            if not retry_allowed:
                return False
            retry_count = int(self._retry_count_by_node.get(source_node_id, 0))
            if retry_count >= self.max_retries_per_chunk:
                self._retry_allowed_by_node[source_node_id] = False
                return False
            previous_dispatch_event_ns = self._last_dispatch_event_time_ns_by_node.get(source_node_id)
            if (
                previous_dispatch_event_ns is not None
                and event_time_ns <= previous_dispatch_event_ns + self._min_retry_progress_ns
            ):
                return False
            self._retry_allowed_by_node[source_node_id] = False
            self._retry_count_by_node[source_node_id] = retry_count + 1
            self._last_dispatch_event_time_ns_by_node[source_node_id] = int(event_time_ns)
            return True
        self._last_chunk_id_by_node[source_node_id] = chunk_id
        self._retry_allowed_by_node[source_node_id] = False
        self._retry_count_by_node[source_node_id] = 0
        self._last_dispatch_event_time_ns_by_node[source_node_id] = int(event_time_ns)
        return True

    def record_dispatch_outcome(
        self,
        *,
        source_node_id: str,
        event_time_ns: int,
        produced_actionable_detection: bool,
        allow_retry_for_non_actionable: bool = True,
    ) -> None:
        """Record whether a chunk dispatch produced an actionable detection.

        Non-actionable outputs (for example, unknown/0.0 confidence) allow a
        bounded retry within the same chunk so early low-context dispatches do
        not suppress later, richer context while still protecting compute.
        """
        chunk_id = int(event_time_ns) // self._stride_ns
        self._last_chunk_id_by_node[source_node_id] = chunk_id
        retry_count = int(self._retry_count_by_node.get(source_node_id, 0))
        self._retry_allowed_by_node[source_node_id] = (
            not produced_actionable_detection
            and allow_retry_for_non_actionable
            and retry_count < self.max_retries_per_chunk
        )
