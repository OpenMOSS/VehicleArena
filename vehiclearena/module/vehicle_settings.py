"""
VehicleSettings - Vehicle global shared state (instance-based).

All vehicle modules that need shared state (volume, sound_channel, speaker, etc.)
receive a reference to the same VehicleSettings instance via set_settings().

Fields:
  - volume: Global speaker volume (0-100)
  - sound_channel: Active audio source (music/video/radio/conversation)
  - unit_system: Distance units (mile/kilometer)
  - timestamp: System clock
  - speaker: Current speaker seat position
  - temperature: Ambient/cabin temperature
  - language: System UI language
  - time_display_format: 12-hour or 24-hour
"""

from typing import Dict, Any


class VehicleSettings:
    def __init__(self):
        self.volume = 50
        self.sound_channel = "music"
        self.unit_system = "mile"
        self.timestamp = "2025-04-13 12:00:00"
        self.speaker = "driver's seat"
        self.temperature = 14
        self.language = "Chinese"
        self.time_display_format = "24-hour-format"

    # ── Presets ──

    @classmethod
    def init1(cls) -> 'VehicleSettings':
        """Music channel, volume 60."""
        instance = cls()
        instance.volume = 60
        instance.sound_channel = "music"
        instance.unit_system = "mile"
        instance.timestamp = "2025-04-13 11:00:00"
        return instance

    @classmethod
    def init2(cls) -> 'VehicleSettings':
        """Video channel, volume 75."""
        instance = cls()
        instance.volume = 75
        instance.sound_channel = "video"
        instance.unit_system = "mile"
        instance.timestamp = "2025-04-13 12:10:00"
        return instance

    @classmethod
    def init4(cls) -> 'VehicleSettings':
        """Radio channel, volume 50."""
        instance = cls()
        instance.volume = 50
        instance.sound_channel = "radio"
        instance.unit_system = "mile"
        instance.timestamp = "2025-04-13 13:00:00"
        return instance

    @classmethod
    def init5(cls) -> 'VehicleSettings':
        """Conversation channel, volume 70."""
        instance = cls()
        instance.volume = 70
        instance.sound_channel = "conversation"
        instance.unit_system = "mile"
        instance.timestamp = "2025-04-13 13:30:00"
        return instance

    @classmethod
    def init6(cls) -> 'VehicleSettings':
        """Music channel, volume 60, driver's seat, temp 15."""
        instance = cls()
        instance.volume = 60
        instance.sound_channel = "music"
        instance.unit_system = "mile"
        instance.timestamp = "2025-04-13 12:00:00"
        instance.speaker = "driver's seat"
        instance.temperature = 15
        return instance

    @classmethod
    def init7(cls) -> 'VehicleSettings':
        """Music channel, second row left, temp 15, Chinese, 24-hour."""
        instance = cls()
        instance.volume = 60
        instance.sound_channel = "music"
        instance.unit_system = "mile"
        instance.timestamp = "2025-04-13 12:00:00"
        instance.speaker = "second row left"
        instance.temperature = 15
        instance.language = "Chinese"
        instance.time_display_format = "24-hour-format"
        return instance
