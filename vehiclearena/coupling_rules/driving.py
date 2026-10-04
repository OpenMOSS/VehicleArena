"""
Coupling Rules — Driving & Navigation
"""

import logging
from event_bus import EventBus
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
# Rule 17: Video Playback Safety
# ──────────────────────────────────────────────
def _register_video_safety_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """
    Playing video while navigation is active (i.e. vehicle is driving) → SOFT warning.
    Full-screen video while navigation active → HARD block (safety critical).
    """

    def check_video_play_during_driving(module_instance, method_name, args, kwargs, vw):
        if vw is None:
            return None

        try:
            nav = vw.navigation
            # If navigation is active, vehicle is likely in motion
            if hasattr(nav, '_is_navigating') and nav._is_navigating:
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.SOFT,
                    message="Navigation is active (vehicle may be in motion). "
                            "Playing video while driving is not recommended for safety."
                )
        except Exception as e:
            logger.debug(f"Video safety check error: {e}")
        return None

    def check_fullscreen_video_during_driving(module_instance, method_name, args, kwargs, vw):
        if vw is None:
            return None

        # Only when turning ON fullscreen
        switch = args[1] if len(args) > 1 else kwargs.get('switch', None)
        if switch is not True:
            return None

        try:
            nav = vw.navigation
            if hasattr(nav, '_is_navigating') and nav._is_navigating:
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.HARD,
                    message="Navigation is active (vehicle is in motion). "
                            "Full-screen video is BLOCKED for driver safety."
                )
        except Exception as e:
            logger.debug(f"Fullscreen video safety check error: {e}")
        return None

    # Soft: any video play while driving
    for method in ["video_download_play", "video_local_play", "video_history_play",
                   "video_favorite_play"]:
        constraint_engine.register(Constraint(
            name=f"video_play_driving_safety_{method}",
            level=ConstraintLevel.SOFT,
            check_fn=check_video_play_during_driving,
            phase="pre",
            target_modules=["video"],
            target_methods=[method]
        ))

    # Hard: fullscreen video while driving
    constraint_engine.register(Constraint(
        name="fullscreen_video_driving_block",
        level=ConstraintLevel.HARD,
        check_fn=check_fullscreen_video_during_driving,
        phase="pre",
        target_modules=["video"],
        target_methods=["video_fullScreenPlay_switch"]
    ))




# ──────────────────────────────────────────────
# Rule 18: Trunk/FrontTrunk + Navigation Safety
# ──────────────────────────────────────────────
def _register_trunk_navigation_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """
    Opening trunk or front trunk while navigation is active (vehicle likely moving)
    → SOFT warning about safety.
    """

    def check_trunk_while_navigating(module_instance, method_name, args, kwargs, vw):
        if vw is None:
            return None

        # Only check when opening
        action = args[1] if len(args) > 1 else kwargs.get('action', None)
        if action != 'open':
            return None

        try:
            nav = vw.navigation
            if hasattr(nav, '_is_navigating') and nav._is_navigating:
                return ConstraintResult(
                    passed=False,
                    level=ConstraintLevel.SOFT,
                    message="Navigation is active (vehicle may be in motion). "
                            "Opening the trunk while driving is not recommended."
                )
        except Exception as e:
            logger.debug(f"Trunk-navigation check error: {e}")
        return None

    constraint_engine.register(Constraint(
        name="trunk_open_while_navigating",
        level=ConstraintLevel.SOFT,
        check_fn=check_trunk_while_navigating,
        phase="pre",
        target_modules=["trunk"],
        target_methods=["carcontrol_trunk_switch"]
    ))

    constraint_engine.register(Constraint(
        name="front_trunk_open_while_navigating",
        level=ConstraintLevel.SOFT,
        check_fn=check_trunk_while_navigating,
        phase="pre",
        target_modules=["frontTrunk"],
        target_methods=["carcontrol_frontTrunk_switch"]
    ))



def register_driving_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """Register all driving & navigation rules."""
    _register_video_safety_rules(event_bus, constraint_engine)
    _register_trunk_navigation_rules(event_bus, constraint_engine)
