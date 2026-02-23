"""
Activity Recognition Module.

Rule-based activity detection combining gesture, object,
and temporal signals to infer what the person is doing.
"""

import time


class ActivityRecognizer:
    """
    Rule-based activity recognizer.
    
    Detects activities like Writing, Waving, Talking on phone,
    Reading, Using laptop, Idle, etc.
    
    Uses temporal smoothing — an activity must persist for 1+ seconds
    before being reported.
    """

    ACTIVITIES = {
        "writing": "Writing",
        "waving": "Waving",
        "phone_call": "Talking on Phone",
        "reading": "Reading",
        "using_laptop": "Using Laptop",
        "pointing": "Pointing",
        "greeting": "Greeting",
        "thinking": "Thinking",
        "presenting": "Presenting",
        "idle": "Idle",
    }

    def __init__(self, persistence_seconds=1.0):
        self.persistence = persistence_seconds
        self._candidate = None
        self._candidate_start = 0
        self._confirmed_activity = "idle"
        self._last_activity = "idle"

    def update(self, object_detections=None, gesture_detections=None,
               emotion_detections=None, ocr_text=""):
        """
        Infer the current activity based on available signals.

        Returns:
            The confirmed activity display name.
        """
        detected_objects = set()
        if object_detections:
            for label, conf, _ in object_detections:
                detected_objects.add(label.lower())

        gesture_name = ""
        if gesture_detections:
            for gest, conf, _ in gesture_detections:
                if conf > 0.5:
                    gesture_name = gest.split("|")[0].strip().lower()
                    break

        emotion = ""
        if emotion_detections:
            for emo, conf, _ in emotion_detections:
                if conf > 0.4:
                    emotion = emo.lower()
                    break

        # ── Rule-based activity inference ──
        activity = "idle"

        # Writing: pointing gesture (drawing) + OCR has text
        if "point" in gesture_name or "count: 1" in gesture_name:
            if ocr_text and len(ocr_text) > 2:
                activity = "writing"
            else:
                activity = "pointing"

        # Waving: open palm gesture
        elif "open palm" in gesture_name or "count: 5" in gesture_name:
            activity = "waving"

        # Greeting: thumbs up
        elif "thumbs up" in gesture_name:
            activity = "greeting"

        # Phone call: cell phone detected + person
        elif "cell phone" in detected_objects and "person" in detected_objects:
            activity = "phone_call"

        # Using laptop: laptop detected + person
        elif "laptop" in detected_objects and "person" in detected_objects:
            activity = "using_laptop"

        # Reading: book detected or person looking down
        elif "book" in detected_objects:
            activity = "reading"

        # Presenting: person + peace/pointing
        elif "peace" in gesture_name or "victory" in gesture_name:
            activity = "presenting"

        # Thinking: neutral face, no gestures
        elif emotion in ("neutral", "sad") and not gesture_name:
            activity = "thinking"

        # ── Temporal smoothing ──
        now = time.time()
        if activity != self._candidate:
            self._candidate = activity
            self._candidate_start = now
        elif (now - self._candidate_start) >= self.persistence:
            self._confirmed_activity = activity

        return self.ACTIVITIES.get(self._confirmed_activity, "Idle")

    def get_activity(self):
        return self.ACTIVITIES.get(self._confirmed_activity, "Idle")

    def draw(self, frame):
        """Draw current activity as a prominent label."""
        import cv2

        activity = self.get_activity()
        h, w = frame.shape[:2]

        # Position at top-center-right
        text = f"Activity: {activity}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx = w - tw - 20
        ty = 30

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (tx - 10, ty - 22), (tx + tw + 10, ty + 8), (20, 30, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(
            frame, text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 150), 2, cv2.LINE_AA,
        )
