"""
Performance tracking utilities for FPS and per-module latency measurement.
"""

import time
from collections import deque
import config


class FPSCounter:
    """
    Rolling average FPS counter using a fixed-size window.
    """

    def __init__(self, window_size=None):
        self.window_size = window_size or config.FPS_ROLLING_WINDOW
        self._timestamps = deque(maxlen=self.window_size)
        self._last_tick = None

    def tick(self):
        """Call once per frame to record a timestamp."""
        now = time.perf_counter()
        if self._last_tick is not None:
            self._timestamps.append(now - self._last_tick)
        self._last_tick = now

    def get_fps(self):
        """Return the rolling average FPS."""
        if not self._timestamps:
            return 0.0
        avg_delta = sum(self._timestamps) / len(self._timestamps)
        return 1.0 / avg_delta if avg_delta > 0 else 0.0


class LatencyTracker:
    """
    Track inference latency for each AI module.
    """

    def __init__(self):
        self._timers = {}
        self._latencies = {}

    def start(self, module_name):
        """Start timing a module."""
        self._timers[module_name] = time.perf_counter()

    def stop(self, module_name):
        """
        Stop timing a module and record the latency.

        Returns:
            Latency in milliseconds.
        """
        if module_name in self._timers:
            elapsed = (time.perf_counter() - self._timers[module_name]) * 1000
            self._latencies[module_name] = elapsed
            del self._timers[module_name]
            return elapsed
        return 0.0

    def get_latency(self, module_name):
        """
        Get the last recorded latency for a module.

        Returns:
            Latency in milliseconds, or 0 if not yet recorded.
        """
        return self._latencies.get(module_name, 0.0)

    def get_all_latencies(self):
        """
        Get all recorded latencies.

        Returns:
            Dict mapping module names to latency in ms.
        """
        return dict(self._latencies)

    def get_summary_lines(self):
        """
        Get formatted latency lines for display.

        Returns:
            List of strings like 'ModuleName: 12.3 ms'.
        """
        lines = []
        for name, ms in self._latencies.items():
            lines.append(f"{name}: {ms:.1f} ms")
        return lines
