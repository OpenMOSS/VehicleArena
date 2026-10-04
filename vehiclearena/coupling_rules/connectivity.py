"""
Coupling Rules — Connectivity
"""

import logging
from event_bus import EventBus, Event, EventPriority
from constraints import (
    ConstraintEngine, Constraint, ConstraintLevel, ConstraintResult,
)

logger = logging.getLogger(__name__)


def _get_coupling_state(constraint_engine):
    """Get per-instance coupling state dict (thread-safe)."""
    if not hasattr(constraint_engine, '_coupling_state'):
        constraint_engine._coupling_state = {}
    return constraint_engine._coupling_state




# ──────────────────────────────────────────────
# Rule 6: Bluetooth Dependency
# ──────────────────────────────────────────────
def _register_bluetooth_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """
    Making a phone call requires Bluetooth to be enabled and connected.
    This is a SOFT constraint that warns the user.
    """

    def check_bluetooth_for_call(module_instance, method_name, args, kwargs, vw):
        """Check if Bluetooth is available before making a call."""
        if vw is None:
            return None

        try:
            bt_dict = vw.bluetooth.to_dict()
            bt_enabled = bt_dict.get("is_enabled", {}).get("value", False)
            bt_connected = bt_dict.get("connection_state", {}).get("value", "DISCONNECTED")

            if not bt_enabled:
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.SOFT,
                    message="Bluetooth is disabled. Phone call may use vehicle speakers instead. "
                            "Enable Bluetooth for hands-free calling."
                )
            elif bt_connected != "CONNECTED":
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.SOFT,
                    message="Bluetooth is enabled but no device is connected. "
                            "Phone call may not work properly without a connected device."
                )
        except Exception as e:
            logger.debug(f"Bluetooth constraint check error: {e}")

        return None

    constraint_engine.register(Constraint(
        name="bluetooth_required_for_call",
        level=ConstraintLevel.SOFT,
        check_fn=check_bluetooth_for_call,
        phase="pre",
        target_modules=["conversation"],
        target_methods=["conversation_phone_call", "conversation_phone_redial"]
    ))

    def on_bluetooth_disconnected(event: Event):
        """When Bluetooth disconnects, if there's an active call, warn about it."""
        vw = constraint_engine._vw
        if vw is None:
            return

        try:
            if vw.conversation.call_state == "active":
                logger.warning(
                    "Bluetooth disconnected during active call! "
                    "Call may be transferred to vehicle speakers."
                )
                return {
                    "warning": "bluetooth_disconnected_during_call",
                    "message": "Bluetooth disconnected during active call."
                }
        except Exception as e:
            logger.debug(f"Bluetooth-call coupling error: {e}")

    event_bus.subscribe("bluetooth.disconnected", on_bluetooth_disconnected, EventPriority.HIGH)



def register_connectivity_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """Register all connectivity rules."""
    _register_bluetooth_rules(event_bus, constraint_engine)

