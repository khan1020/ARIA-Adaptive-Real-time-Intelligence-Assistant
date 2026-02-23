"""
AI Dashboard / HUD — Professional heads-up display overlay.

Shows real-time detection log, confidence bars, object counts,
and FPS sparkline graph.
"""

import cv2
import time
import config


class AIDashboard:
    """
    Real-time AI analytics dashboard overlay.
    
    Components:
    - Detection log: scrolling list of recent detections
    - Confidence bars: visual bars per detection
    - Object counts: running tally of detected classes
    - FPS sparkline: mini-graph of FPS over time
    """

    def __init__(self, max_log_entries=8, max_fps_history=60):
        self.log = []  # List of (timestamp, category_icon, text, confidence)
        self.max_log = max_log_entries
        self.fps_history = []
        self.max_fps_history = max_fps_history
        self.object_counts = {}
        self.total_detections = 0

    def add_detection(self, category, text, confidence=0.0):
        """Add a detection to the log."""
        now = time.strftime("%H:%M:%S")
        icons = {
            "object": "[OBJ]",
            "emotion": "[EMO]",
            "gesture": "[GES]",
            "ocr": "[OCR]",
            "activity": "[ACT]",
            "relation": "[REL]",
            "mesh": "[MESH]",
        }
        icon = icons.get(category, "[???]")
        self.log.append((now, icon, text[:35], confidence))
        if len(self.log) > self.max_log:
            self.log.pop(0)
        self.total_detections += 1

    def update_fps(self, fps):
        """Record FPS for sparkline."""
        self.fps_history.append(fps)
        if len(self.fps_history) > self.max_fps_history:
            self.fps_history.pop(0)

    def update_counts(self, counts_dict):
        """Update object counts from tracker."""
        self.object_counts = counts_dict

    def log_detections(self, category, detections):
        """Batch log detections from a module."""
        for item in detections[:2]:  # Max 2 per frame per category
            if len(item) >= 2:
                label = item[0] if isinstance(item[0], str) else str(item[0])
                conf = float(item[1]) if len(item) > 1 else 0.0
                self.add_detection(category, label, conf)

    def draw(self, frame):
        """Draw the full AI dashboard on the frame."""
        h, w = frame.shape[:2]

        # Dashboard on the left side, below the info panel
        panel_x = 5
        panel_y = 180
        panel_w = 280
        panel_h = h - panel_y - 50

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (20, 20, 30), -1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Border
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (100, 200, 255), 1,
        )

        # Title
        cv2.putText(
            frame, "AI DASHBOARD",
            (panel_x + 8, panel_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (100, 200, 255), 1, cv2.LINE_AA,
        )

        y_offset = panel_y + 22

        # ── FPS Sparkline ──
        y_offset += 18
        self._draw_sparkline(frame, panel_x + 8, y_offset, panel_w - 20, 30)
        y_offset += 38

        # ── Object Counts ──
        if self.object_counts:
            cv2.putText(
                frame, "Object Counts:",
                (panel_x + 8, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (180, 180, 180), 1, cv2.LINE_AA,
            )
            y_offset += 16
            for name, count in list(self.object_counts.items())[:5]:
                cv2.putText(
                    frame, f"  {name}: {count}",
                    (panel_x + 8, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (200, 255, 200), 1, cv2.LINE_AA,
                )
                y_offset += 14
            y_offset += 6

        # ── Total Detections ──
        cv2.putText(
            frame, f"Total Detections: {self.total_detections}",
            (panel_x + 8, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
            (200, 200, 100), 1, cv2.LINE_AA,
        )
        y_offset += 20

        # ── Detection Log ──
        cv2.putText(
            frame, "Detection Log:",
            (panel_x + 8, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            (180, 180, 180), 1, cv2.LINE_AA,
        )
        y_offset += 5

        for timestamp, icon, text, conf in reversed(self.log):
            y_offset += 15
            if y_offset > panel_y + panel_h - 10:
                break

            # Confidence bar
            bar_w = int(conf * 40)
            bar_color = (0, 255, 0) if conf > 0.7 else (0, 200, 255) if conf > 0.4 else (0, 100, 255)
            cv2.rectangle(
                frame,
                (panel_x + 8, y_offset - 8),
                (panel_x + 8 + bar_w, y_offset - 2),
                bar_color, -1,
            )

            # Log text
            log_text = f"{icon} {text}"
            cv2.putText(
                frame, log_text,
                (panel_x + 52, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                (220, 220, 220), 1, cv2.LINE_AA,
            )

    def _draw_sparkline(self, frame, x, y, w, h):
        """Draw a mini FPS graph."""
        if len(self.fps_history) < 2:
            return

        cv2.putText(
            frame, "FPS",
            (x, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
            (150, 150, 150), 1, cv2.LINE_AA,
        )

        # Draw background
        cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 50), -1)

        max_fps = max(self.fps_history) if self.fps_history else 30
        min_fps = min(self.fps_history) if self.fps_history else 0
        fps_range = max(max_fps - min_fps, 1)

        n = len(self.fps_history)
        step = w / max(n - 1, 1)

        points = []
        for i, fps in enumerate(self.fps_history):
            px = int(x + i * step)
            py = int(y + h - ((fps - min_fps) / fps_range) * h)
            points.append((px, py))

        for i in range(1, len(points)):
            color = (0, 255, 100) if self.fps_history[i] > 15 else (0, 100, 255)
            cv2.line(frame, points[i - 1], points[i], color, 1, cv2.LINE_AA)

        # Current FPS label
        current = self.fps_history[-1] if self.fps_history else 0
        cv2.putText(
            frame, f"{current:.0f}",
            (x + w - 25, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
            (0, 255, 100), 1, cv2.LINE_AA,
        )
