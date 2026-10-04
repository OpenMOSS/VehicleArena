"""
Layer 1: EventBus - Publish-Subscribe Event System
Provides inter-module communication through an event-driven architecture.
"""

import threading
from enum import IntEnum
from typing import Callable, Dict, List, Any, Optional
from collections import defaultdict


class EventPriority(IntEnum):
    """Event handler priority levels. Lower number = higher priority."""
    CRITICAL = 0    # Safety-critical handlers (e.g., collision detection)
    HIGH = 10       # Constraint checks
    NORMAL = 50     # Standard business logic
    LOW = 100       # Logging, analytics


class Event:
    """Represents an event in the system."""

    def __init__(self, event_type: str, source: str, data: Optional[Dict[str, Any]] = None):
        """
        Args:
            event_type: Event type identifier (e.g., "call.started", "window.opened")
            source: Source module name (e.g., "conversation", "window")
            data: Optional dictionary with event-specific data
        """
        self.event_type = event_type
        self.source = source
        self.data = data or {}
        self.cancelled = False      # If True, downstream handlers are skipped
        self.results = []           # Collected results from handlers

    def cancel(self):
        """Cancel this event, preventing further handler execution."""
        self.cancelled = True

    def __repr__(self):
        return f"Event(type={self.event_type}, source={self.source}, data={self.data})"


class EventBus:
    """
    Central event bus for publish-subscribe communication between modules.
    Thread-safe, priority-based, synchronous dispatch.
    """

    def __init__(self):
        self._handlers: Dict[str, List[tuple]] = defaultdict(list)  # event_type -> [(priority, handler)]
        self._sorted: Dict[str, bool] = defaultdict(lambda: True)   # Track if handlers are sorted
        self._lock = threading.Lock()
        self._enabled = True

    def __deepcopy__(self, memo):
        """Support deep-copying by creating a fresh EventBus (new lock).

        Handlers are NOT copied — the copy is a blank bus.  This is
        intentional: GT evaluation VWs don't need coupling rules.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result._handlers = defaultdict(list)
        result._sorted = defaultdict(lambda: True)
        result._lock = threading.Lock()
        result._enabled = self._enabled
        return result

    def subscribe(self, event_type: str, handler: Callable[[Event], None],
                  priority: int = EventPriority.NORMAL):
        """
        Subscribe a handler to an event type.

        Args:
            event_type: Event type to listen for (supports wildcard '*' suffix, e.g., "call.*")
            handler: Callable that accepts an Event object
            priority: Handler priority (lower = called first)
        """
        with self._lock:
            self._handlers[event_type].append((priority, handler))
            self._sorted[event_type] = False

    def unsubscribe(self, event_type: str, handler: Callable[[Event], None]):
        """Remove a handler from an event type."""
        with self._lock:
            self._handlers[event_type] = [
                (p, h) for p, h in self._handlers[event_type] if h != handler
            ]

    def publish(self, event: Event) -> Event:
        """
        Publish an event synchronously. Handlers are called in priority order.

        Args:
            event: The Event to publish

        Returns:
            The Event object (may have been modified by handlers)
        """
        if not self._enabled:
            return event

        handlers = self._collect_handlers(event.event_type)

        for priority, handler in handlers:
            if event.cancelled:
                break
            try:
                result = handler(event)
                if result is not None:
                    event.results.append(result)
            except Exception as e:
                # Log but don't crash - coupling should be resilient
                import logging
                logging.getLogger(__name__).warning(
                    f"Event handler error for {event.event_type}: {e}"
                )

        return event

    def _collect_handlers(self, event_type: str) -> List[tuple]:
        """Collect all matching handlers for an event type, including wildcard matches."""
        with self._lock:
            all_handlers = []

            # Exact match
            if event_type in self._handlers:
                if not self._sorted[event_type]:
                    self._handlers[event_type].sort(key=lambda x: x[0])
                    self._sorted[event_type] = True
                all_handlers.extend(self._handlers[event_type])

            # Wildcard match: "call.*" matches "call.started", "call.ended", etc.
            parts = event_type.split('.')
            for i in range(len(parts)):
                wildcard = '.'.join(parts[:i+1]) + '.*'
                if wildcard in self._handlers and wildcard != event_type:
                    if not self._sorted[wildcard]:
                        self._handlers[wildcard].sort(key=lambda x: x[0])
                        self._sorted[wildcard] = True
                    all_handlers.extend(self._handlers[wildcard])

            # Global wildcard
            if '*' in self._handlers:
                if not self._sorted['*']:
                    self._handlers['*'].sort(key=lambda x: x[0])
                    self._sorted['*'] = True
                all_handlers.extend(self._handlers['*'])

            # Sort all collected handlers by priority
            all_handlers.sort(key=lambda x: x[0])
            return all_handlers

    def enable(self):
        """Enable event dispatching."""
        self._enabled = True

    def disable(self):
        """Disable event dispatching (useful during initialization/testing)."""
        self._enabled = False

    def clear(self):
        """Remove all handlers."""
        with self._lock:
            self._handlers.clear()
            self._sorted.clear()
