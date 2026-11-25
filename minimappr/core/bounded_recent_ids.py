"""A bounded, insertion-ordered set of recently seen ids (oldest evicted first)."""

from __future__ import annotations

from collections import OrderedDict


class BoundedRecentIds:
    def __init__(self, max_entries: int) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._max_entries = max_entries
        self._ids: OrderedDict[str, None] = OrderedDict()

    def add_if_new(self, item_id: str) -> bool:
        """Remember ``item_id``; True when it was not already remembered."""
        if item_id in self._ids:
            return False
        self._ids[item_id] = None
        while len(self._ids) > self._max_entries:
            self._ids.popitem(last=False)
        return True

    def __contains__(self, item_id: object) -> bool:
        return item_id in self._ids

    def __len__(self) -> int:
        return len(self._ids)
