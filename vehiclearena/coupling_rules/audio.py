"""
Coupling Rules — Audio & Media
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
# Rule 1: Audio Ducking / Channel Conflict
# ──────────────────────────────────────────────
def _register_audio_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """
    When a call starts, media volume is ducked (lowered).
    When a call ends, media volume is restored.
    """

    def on_call_started(event: Event):
        """Duck media volume when a call starts."""
        vw = constraint_engine._vw
        if vw is None:
            return

        current_vol = vw.settings.volume
        current_channel = vw.settings.sound_channel

        # Save pre-call state in per-instance coupling state for restoration on call end
        state = _get_coupling_state(constraint_engine)
        state['pre_call_volume'] = current_vol
        state['pre_call_channel'] = current_channel

        # If media is playing, duck to 30% of current volume
        if current_channel in ('music', 'video', 'radio'):
            ducked_volume = max(10, int(current_vol * 0.3))
            vw.settings.volume = ducked_volume
            logger.info(
                f"Audio ducking: {current_channel} volume {current_vol} -> {ducked_volume} "
                f"due to incoming call"
            )

    def on_call_ended(event: Event):
        """Restore media volume when a call ends."""
        vw = constraint_engine._vw
        if vw is None:
            return

        state = _get_coupling_state(constraint_engine)
        pre_vol = state.pop('pre_call_volume', None)
        pre_channel = state.pop('pre_call_channel', None)

        if pre_vol is not None:
            vw.settings.volume = pre_vol
            logger.info(f"Audio restored: volume -> {pre_vol}")
        if pre_channel is not None:
            vw.settings.sound_channel = pre_channel
            logger.info(f"Audio channel restored: -> {pre_channel}")

    event_bus.subscribe("call.started", on_call_started, EventPriority.HIGH)
    event_bus.subscribe("call.ended", on_call_ended, EventPriority.HIGH)




# ──────────────────────────────────────────────
# Rule 2: Call Interruption (pause media on call)
# ──────────────────────────────────────────────
def _register_call_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """
    When a call starts, pause music/video playback.
    When a call ends, optionally resume playback.
    """

    def on_call_started_pause_media(event: Event):
        """Pause music and video when a call starts."""
        vw = constraint_engine._vw
        if vw is None:
            return

        # Record playback state in per-instance state before pausing
        state = _get_coupling_state(constraint_engine)
        music = vw._get_module("music")
        video = vw._get_module("video")
        state['music_was_playing'] = bool(
            music is not None and music._is_playing)
        state['video_was_playing'] = bool(
            video is not None and video._is_playing)

        if music is not None and music._is_playing:
            music._is_playing = False
            logger.info("Call interruption: Music paused")

        if video is not None and video._is_playing:
            video._is_playing = False
            logger.info("Call interruption: Video paused")

    def on_call_ended_resume_media(event: Event):
        """Resume music/video when a call ends."""
        vw = constraint_engine._vw
        if vw is None:
            return

        state = _get_coupling_state(constraint_engine)
        music = vw._get_module("music")
        video = vw._get_module("video")
        if state.pop('music_was_playing', False) and music is not None:
            music._is_playing = True
            logger.info("Call ended: Music resumed")

        if state.pop('video_was_playing', False) and video is not None:
            video._is_playing = True
            logger.info("Call ended: Video resumed")

    event_bus.subscribe("call.started", on_call_started_pause_media, EventPriority.NORMAL)
    event_bus.subscribe("call.ended", on_call_ended_resume_media, EventPriority.NORMAL)




def register_audio_rules(event_bus: EventBus, constraint_engine: ConstraintEngine):
    """Register all audio & media rules."""
    _register_audio_rules(event_bus, constraint_engine)
    _register_call_rules(event_bus, constraint_engine)
