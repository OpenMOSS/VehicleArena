"""
SessionHistory — Searchable cross-wake event/action index.

The LLM harness no longer asks a model to summarize conversation history and
does not auto-inject records from this store.  Each wake writes structured
event, passenger-message and tool-action metadata.  ``search(keyword)`` is a
bounded, explicit lookup used only when the agent asks for old information.

Usage:
    history = SessionHistory()
    history.record_wake(..., action_log=[...])
    history.search("wiper")      # agent tool call
"""

from typing import Any, Dict, List, Optional


# ── Action log entry ───────────────────────────────────────────

class ActionLogEntry:
    """One timestamped tool call record.

    Attributes:
        timestamp:      Fine-grained time string, e.g. "20m10s".
        func_name:      Full tool name, e.g. "wiper__carcontrol_wiperBlade_switch".
        arguments:      Call arguments as compact string.
        result_snippet: Short result (truncated to ~120 chars).
    """

    def __init__(self, timestamp: str, func_name: str,
                 arguments: str = "", result_snippet: str = ""):
        self.timestamp = timestamp
        self.func_name = func_name
        self.arguments = arguments
        self.result_snippet = result_snippet

    def to_line(self, event_id: str = "") -> str:
        """One-line representation for search results."""
        prefix = f"[{event_id}] " if event_id else ""
        args_part = f"({self.arguments})" if self.arguments else "()"
        result_part = f" → {self.result_snippet}" if self.result_snippet else ""
        return f"{prefix}t={self.timestamp} {self.func_name}{args_part}{result_part}"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "func_name": self.func_name,
            "arguments": self.arguments,
            "result_snippet": self.result_snippet,
        }

    def matches(self, keyword: str) -> bool:
        """Case-insensitive keyword match against all fields."""
        kw = keyword.lower()
        return (kw in self.func_name.lower()
                or kw in self.arguments.lower()
                or kw in self.result_snippet.lower())


# ── Memory record ──────────────────────────────────────────────

class MemoryRecord:
    """A single indexed wake record.

    Attributes:
        event_id:           Unique event identifier (e.g. "evt_001").
        time_s:             Simulation time in seconds when the event occurred.
        event_description:  Brief description of the environment event.
        action_log:         Timestamped tool call log (searchable).
        actions_taken:      Raw action strings for eval.
        passenger_messages: Passenger requests at this tick (searchable).
    """

    def __init__(self, event_id: str, time_s: float = 0.0,
                 event_description: str = "",
                 action_log: List[ActionLogEntry] = None,
                 actions_taken: List[str] = None,
                 passenger_messages: List[str] = None):
        self.event_id = event_id
        self.time_s = time_s
        self.event_description = event_description
        self.action_log = action_log or []
        self.actions_taken = actions_taken or []
        self.passenger_messages = passenger_messages or []

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "time_s": self.time_s,
            "event_description": self.event_description,
            "actions_taken": self.actions_taken,
            "action_log": [a.to_dict() for a in self.action_log],
            "passenger_messages": self.passenger_messages,
        }

    def __repr__(self):
        return (f"MemoryRecord(event_id={self.event_id!r}, "
                f"t={self.time_s}s, "
                f"log={len(self.action_log)})")


# ── Session history store ─────────────────────────────────────

class SessionHistory:
    """Cross-event index queried explicitly through ``memory_search``."""

    def __init__(self):
        self._records: List[MemoryRecord] = []
        self._by_id: Dict[str, MemoryRecord] = {}
        self._counter = 0

    # ── Write ───────────────────────────────────────────────────

    def record_wake(self, time_s: float = 0.0,
                    event_description: str = "",
                    actions_taken: List[str] = None,
                    action_log: List[ActionLogEntry] = None,
                    event_id: str = None,
                    passenger_messages: List[str] = None) -> str:
        """Append one structured wake interaction to the history index.

        Args:
            time_s:             Simulation time in seconds when the event occurred.
            event_description:  Brief description of the environment change.
            actions_taken:      Raw action strings (for eval).
            action_log:         Timestamped tool call entries.
            event_id:           Explicit ID, or auto-generated ``evt_NNN``.
            passenger_messages: Passenger request strings at this tick.

        Returns:
            event_id of the new record.
        """
        self._counter += 1
        if event_id is None:
            event_id = f"evt_{self._counter:03d}"

        record = MemoryRecord(
            event_id=event_id,
            time_s=time_s,
            event_description=event_description,
            action_log=action_log or [],
            actions_taken=actions_taken or [],
            passenger_messages=passenger_messages or [],
        )
        self._records.append(record)
        self._by_id[event_id] = record
        return event_id

    # ── Read (agent-callable) ──────────────────────────────────

    def search(
        self,
        keyword: str,
        scope: str = "all",
        time_from: int = None,
        time_to: int = None,
        event_id: str = None,
        context: int = 0,
        limit: int = 30,
    ) -> str:
        """Grep-like search across driving session memory.

        Searches across two entry types per record (case-insensitive):
          - ``passenger``: passenger request messages.
          - ``action``: tool call logs (func_name, arguments, result).

        Supports scope filtering, time-range filtering, event-level
        pinpointing, and context expansion (like grep -C).

        Args:
            keyword:   Search pattern (case-insensitive substring match).
            scope:     Entry types to search: ``"all"`` (default),
                       ``"passenger"``, or ``"action"``.
            time_from: Only search records at t >= this (seconds).
            time_to:   Only search records at t <= this (seconds).
            event_id:  Only search within a specific event (e.g. "evt_003").
            context:   Show N surrounding entries before/after each match
                       within the same record (like grep -C). Default 0.
            limit:     Maximum result lines. Default 30.

        Returns:
            Formatted multi-line string of matching (and context) entries,
            or a "no matches" message.
        """
        kw = keyword.lower()
        output_lines: List[str] = []
        match_count = 0

        for rec in self._records:
            # ── Record-level filters ──
            if event_id is not None and rec.event_id != event_id:
                continue
            if time_from is not None and rec.time_s < time_from:
                continue
            if time_to is not None and rec.time_s > time_to:
                continue

            # ── Build flat entry list for this record ──
            prefix = f"[{rec.event_id}] t={rec.time_s:.0f}s"
            entries: List[tuple] = []  # (formatted_line, searchable_text)

            if scope in ("all", "passenger"):
                for msg in rec.passenger_messages:
                    entries.append((
                        f'{prefix} [Passenger] "{msg}"',
                        msg.lower(),
                    ))
            if scope in ("all", "action"):
                for entry in rec.action_log:
                    searchable = (f"{entry.func_name} {entry.arguments} "
                                  f"{entry.result_snippet}").lower()
                    entries.append((entry.to_line(rec.event_id), searchable))

            if not entries:
                continue

            # ── Find matching indices and expand context ──
            matched_indices = set()
            for i, (_, searchable) in enumerate(entries):
                if kw in searchable:
                    lo = max(0, i - context)
                    hi = min(len(entries), i + context + 1)
                    for j in range(lo, hi):
                        matched_indices.add(j)

            if not matched_indices:
                continue

            # ── Emit lines with separator between context groups ──
            prev_idx = -2
            for i in sorted(matched_indices):
                if prev_idx >= 0 and i > prev_idx + 1:
                    output_lines.append("  ...")
                output_lines.append(entries[i][0])
                prev_idx = i
                match_count += 1
                if match_count >= limit:
                    break

            if match_count >= limit:
                break

        if not output_lines:
            return f"No matches found for '{keyword}'."
        result = "\n".join(output_lines)
        if match_count >= limit:
            result += f"\n... (truncated at {limit} results)"
        return result

    def recall(self, event_id: str) -> Optional[dict]:
        """Recall a specific record by event_id (summary + action_log).

        Returns:
            Record dict, or None if not found.
        """
        rec = self._by_id.get(event_id)
        return rec.to_dict() if rec else None

    def recall_recent(self, n: int = 3) -> List[dict]:
        """Recall the most recent n memory records.

        Returns:
            List of record dicts, most recent last.
        """
        return [r.to_dict() for r in self._records[-n:]]

    def recall_all(self) -> List[dict]:
        """Recall all memory records.

        Returns:
            List of all record dicts in chronological order.
        """
        return [r.to_dict() for r in self._records]

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def event_ids(self) -> List[str]:
        """All event IDs in chronological order."""
        return [r.event_id for r in self._records]

    def clear(self):
        """Clear all memory records."""
        self._records.clear()
        self._by_id.clear()
        self._counter = 0

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def make_timestamp(simulation_time_s: float) -> str:
        """Format an authoritative simulation time for a memory entry.

        Args:
            simulation_time_s: World-clock time at which the action ran.

        Returns:
            For example, ``120`` becomes ``"2m00s"``.

        Tool calls made during one wake do not advance the physical clock, so
        every action in that wake deliberately receives the same timestamp.
        """
        total_seconds = max(0, int(round(simulation_time_s)))
        total_minutes, remaining_seconds = divmod(total_seconds, 60)
        return f"{total_minutes}m{remaining_seconds:02d}s"
