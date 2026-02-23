"""
Drowsiness / Attention Detection Module.

Uses Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)
computed from MediaPipe face mesh landmarks to detect:
  - Drowsiness (eyes closed for 2+ seconds)
  - Yawning (mouth wide open)
  - Attention level (based on eye openness)

Triggers visual + voice alerts when drowsiness is detected.
"""

import cv2
import math
import time


class DrowsinessDetector:
    """
    Real-time drowsiness and attention monitoring.

    Uses MediaPipe FaceLandmarker output (478 landmarks) to compute
    Eye Aspect Ratio and Mouth Aspect Ratio for state detection.
    """

    # MediaPipe face mesh landmark indices for EAR
    # Left eye: top(159), bottom(145), left(33), right(133), top2(158), bottom2(153)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    # Right eye: top(386), bottom(374), left(362), right(263), top2(385), bottom2(380)
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    # Mouth landmarks for MAR
    # Top lip center(13), bottom lip center(14), left(78), right(308), top2(82), bottom2(87)
    MOUTH = [78, 82, 13, 308, 87, 14]

    # Thresholds
    EAR_THRESHOLD = 0.22       # Below this = eyes closed
    MAR_THRESHOLD = 0.65       # Above this = yawning
    DROWSY_TIME_THRESHOLD = 2.0  # Seconds of closed eyes = drowsy
    YAWN_TIME_THRESHOLD = 1.5    # Seconds of open mouth = yawning

    def __init__(self):
        self._eyes_closed_start = None
        self._yawn_start = None
        self._is_drowsy = False
        self._is_yawning = False
        self._current_ear = 0.0
        self._current_mar = 0.0
        self._attention_level = 100  # 0-100%
        self._alert_triggered = False

    def _distance(self, p1, p2):
        """2D distance between two landmarks."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _compute_ear(self, landmarks, eye_indices):
        """
        Compute Eye Aspect Ratio (EAR).
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        p = [landmarks[i] for i in eye_indices]
        # Vertical distances
        v1 = self._distance(p[1], p[5])  # top to bottom
        v2 = self._distance(p[2], p[4])  # top2 to bottom2
        # Horizontal distance
        h = self._distance(p[0], p[3])

        if h == 0:
            return 0.3
        return (v1 + v2) / (2.0 * h)

    def _compute_mar(self, landmarks, mouth_indices):
        """
        Compute Mouth Aspect Ratio (MAR).
        MAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        p = [landmarks[i] for i in mouth_indices]
        v1 = self._distance(p[1], p[5])
        v2 = self._distance(p[2], p[4])
        h = self._distance(p[0], p[3])

        if h == 0:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def update(self, face_landmarks):
        """
        Update drowsiness state from face mesh landmarks.

        Args:
            face_landmarks: MediaPipe face landmarks (478 points).
                            Pass None if no face detected.

        Returns:
            Dict with drowsiness state info.
        """
        if face_landmarks is None:
            self._eyes_closed_start = None
            self._yawn_start = None
            return self._get_state()

        lms = face_landmarks
        now = time.time()

        # ── Compute EAR ──
        left_ear = self._compute_ear(lms, self.LEFT_EYE)
        right_ear = self._compute_ear(lms, self.RIGHT_EYE)
        self._current_ear = (left_ear + right_ear) / 2.0

        # ── Compute MAR ──
        self._current_mar = self._compute_mar(lms, self.MOUTH)

        # ── Eye state ──
        if self._current_ear < self.EAR_THRESHOLD:
            if self._eyes_closed_start is None:
                self._eyes_closed_start = now
            elif (now - self._eyes_closed_start) >= self.DROWSY_TIME_THRESHOLD:
                self._is_drowsy = True
                self._alert_triggered = True
        else:
            self._eyes_closed_start = None
            self._is_drowsy = False

        # ── Yawn state ──
        if self._current_mar > self.MAR_THRESHOLD:
            if self._yawn_start is None:
                self._yawn_start = now
            elif (now - self._yawn_start) >= self.YAWN_TIME_THRESHOLD:
                self._is_yawning = True
        else:
            self._yawn_start = None
            self._is_yawning = False

        # ── Attention level ──
        # Based on EAR: fully open ~0.35, closed ~0.15
        ear_normalized = max(0, min(1, (self._current_ear - 0.15) / 0.20))
        self._attention_level = int(ear_normalized * 100)

        return self._get_state()

    def _get_state(self):
        return {
            "drowsy": self._is_drowsy,
            "yawning": self._is_yawning,
            "ear": self._current_ear,
            "mar": self._current_mar,
            "attention": self._attention_level,
            "alert": self._alert_triggered,
        }

    def should_alert(self):
        """Check if a voice alert should be triggered (one-shot)."""
        if self._alert_triggered:
            self._alert_triggered = False
            return True
        return False

    def draw(self, frame):
        """Draw drowsiness monitoring overlay."""
        h, w = frame.shape[:2]

        # Panel position — top left area below other info
        px = 10
        py = 50

        # ── EAR / MAR gauges ──
        ear_text = f"EAR: {self._current_ear:.2f}"
        mar_text = f"MAR: {self._current_mar:.2f}"
        attn_text = f"Attention: {self._attention_level}%"

        ear_color = (0, 255, 0) if self._current_ear >= self.EAR_THRESHOLD else (0, 0, 255)
        mar_color = (0, 255, 0) if self._current_mar < self.MAR_THRESHOLD else (0, 165, 255)

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 175, py), (w - 5, py + 82), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, ear_text, (w - 170, py + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, ear_color, 1, cv2.LINE_AA)
        cv2.putText(frame, mar_text, (w - 170, py + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, mar_color, 1, cv2.LINE_AA)

        # Attention bar
        cv2.putText(frame, attn_text, (w - 170, py + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        bar_w = 100
        bar_filled = int(self._attention_level / 100 * bar_w)
        bar_color = (0, 255, 0) if self._attention_level > 60 else (0, 165, 255) if self._attention_level > 30 else (0, 0, 255)
        cv2.rectangle(frame, (w - 170, py + 62), (w - 170 + bar_w, py + 72), (40, 40, 40), -1)
        cv2.rectangle(frame, (w - 170, py + 62), (w - 170 + bar_filled, py + 72), bar_color, -1)

        # ── DROWSY ALERT ──
        if self._is_drowsy:
            alert_text = "!! DROWSY - WAKE UP !!"
            (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            ax = (w - tw) // 2
            ay = h // 2

            # Flashing red background
            flash = int(time.time() * 4) % 2 == 0
            if flash:
                overlay2 = frame.copy()
                cv2.rectangle(overlay2, (ax - 20, ay - 40), (ax + tw + 20, ay + 15), (0, 0, 180), -1)
                cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

            cv2.putText(frame, alert_text, (ax, ay),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 3, cv2.LINE_AA)

        # ── YAWN indicator ──
        if self._is_yawning:
            cv2.putText(frame, "* YAWNING *", (w // 2 - 80, h // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 165, 255), 2, cv2.LINE_AA)
