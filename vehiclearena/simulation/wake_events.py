"""Unified, deterministic event model for LLM wake-up delivery."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional
class WakePriority(IntEnum):
    INFO = 10
    NORMAL = 20
    IMPORTANT = 30
    CRITICAL = 40


@dataclass(frozen=True)
class WakeEvent:
    event_id: str
    sequence: int
    event_type: str
    entity_id: str
    entity_type: str
    occurred_at_s: float
    detected_at_s: float
    priority: WakePriority = WakePriority.NORMAL
    source: str = "simulation"
    dedupe_key: str = ""
    state: str = "occurred"
    details: Dict[str, Any] = field(default_factory=dict)
    related_entities: List[str] = field(default_factory=list)
    def as_dict(self, delivered_at_s: Optional[float] = None) -> dict:
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "occurred_at_s": self.occurred_at_s,
            "detected_at_s": self.detected_at_s,
            "delivered_at_s": delivered_at_s,
            "priority": self.priority.name.lower(),
            "source": self.source,
            "dedupe_key": self.dedupe_key,
            "state": self.state,
            "details": dict(self.details),
            "related_entities": list(self.related_entities),
        }


class WakeEventBroker:
    """Tracks continuous event states and suppresses unchanged repetitions."""

    def __init__(self):
        self._levels: Dict[str, str] = {}
        self._last_emit_s: Dict[str, float] = {}
        self._last_clear_s: Dict[str, float] = {}
        self._sequence = 0
        self.log: List[WakeEvent] = []
        self.delivery_log: List[Dict[str, Any]] = []

    def _next_identity(self) -> tuple[str, int]:
        self._sequence += 1
        return f"wake-{self._sequence:012d}", self._sequence

    def discrete(
        self, event_type: str, entity_id: str, entity_type: str,
        time_s: float, *, priority: WakePriority = WakePriority.NORMAL,
        source: str = "simulation", details: Optional[dict] = None,
        related_entities: Optional[List[str]] = None,
        detected_at_s: Optional[float] = None,
    ) -> WakeEvent:
        event_id, sequence = self._next_identity()
        event = WakeEvent(
            event_id=event_id, sequence=sequence,
            event_type=event_type, entity_id=entity_id,
            entity_type=entity_type, occurred_at_s=time_s,
            detected_at_s=(
                time_s if detected_at_s is None else detected_at_s),
            priority=priority, source=source, state="occurred",
            details=details or {},
            related_entities=related_entities or [])
        self.log.append(event)
        return event

    def transition(
        self, event_type: str, entity_id: str, entity_type: str,
        time_s: float, level: str, *, dedupe_key: str,
        priority: WakePriority = WakePriority.NORMAL,
        source: str = "simulation", details: Optional[dict] = None,
        related_entities: Optional[List[str]] = None,
        repeat_after_s: Optional[float] = None,
        reenter_cooldown_s: Optional[float] = None,
        detected_at_s: Optional[float] = None,
    ) -> Optional[WakeEvent]:
        previous = self._levels.get(dedupe_key, "clear")
        # A noisy threshold can alternate clear/caution on adjacent physics
        # frames.  Keep clear observable, but suppress a non-critical
        # re-entry until the caller's simulation-time cooldown expires.  The
        # level is deliberately left clear while suppressed so it can be
        # reconsidered on the next frame.
        if (previous == "clear" and level != "clear"
                and reenter_cooldown_s is not None
                and time_s - self._last_clear_s.get(
                    dedupe_key, float("-inf"))
                < max(0.0, float(reenter_cooldown_s))):
            return None
        changed = previous != level
        repeated = (
            repeat_after_s is not None and level != "clear"
            and time_s - self._last_emit_s.get(
                dedupe_key, float("-inf")) >= repeat_after_s)
        if not changed and not repeated:
            return None
        self._levels[dedupe_key] = level
        if previous == "clear" and level != "clear":
            state = "entered"
        elif level == "clear":
            state = "cleared"
        else:
            state = "updated"
        event_id, sequence = self._next_identity()
        event = WakeEvent(
            event_id=event_id, sequence=sequence,
            event_type=event_type, entity_id=entity_id,
            entity_type=entity_type, occurred_at_s=time_s,
            detected_at_s=(
                time_s if detected_at_s is None else detected_at_s),
            priority=priority, source=source,
            dedupe_key=dedupe_key, state=state,
            details={"previous_level": previous, "level": level,
                     **(details or {})},
            related_entities=related_entities or [])
        self._last_emit_s[dedupe_key] = time_s
        self.log.append(event)
        if level == "clear":
            # The event log is the history. Keeping every cleared key in the
            # live state table would leak memory in long, high-churn worlds.
            self._levels.pop(dedupe_key, None)
            self._last_emit_s.pop(dedupe_key, None)
            self._last_clear_s[dedupe_key] = time_s
        else:
            self._last_clear_s.pop(dedupe_key, None)
        return event

    def record_delivery(
        self,
        entity_id: str,
        events: List[WakeEvent],
        delivered_at_s: float,
        *,
        status: str,
        actions: Optional[List[str]] = None,
        error: str = "",
    ) -> None:
        """Audit one callback attempt without mutating frozen events."""
        self.delivery_log.append({
            "entity_id": entity_id,
            "event_ids": [event.event_id for event in events],
            "delivered_at_s": delivered_at_s,
            "status": status,
            "actions": list(actions or []),
            "error": error,
        })


def group_wake_events(events: List[WakeEvent]) -> Dict[str, List[WakeEvent]]:
    grouped: Dict[str, List[WakeEvent]] = {}
    for event in events:
        grouped.setdefault(event.entity_id, []).append(event)
    for items in grouped.values():
        items.sort(key=lambda item: (
            -int(item.priority), item.occurred_at_s, item.sequence))
    return grouped
