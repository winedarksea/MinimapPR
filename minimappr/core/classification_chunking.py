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

    def __post_init__(self) -> None:
        self._stride_ns = max(1, int(self.stride_seconds * 1_000_000_000))

    def should_dispatch(self, *, source_node_id: str, event_time_ns: int) -> bool:
        chunk_id = int(event_time_ns) // self._stride_ns
        previous_chunk_id = self._last_chunk_id_by_node.get(source_node_id)
        if previous_chunk_id == chunk_id:
            return False
        self._last_chunk_id_by_node[source_node_id] = chunk_id
        return True
