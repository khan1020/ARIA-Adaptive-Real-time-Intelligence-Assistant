"""
Scene Description Module — Natural Language Generation.

Combines outputs from all detection modules to generate
human-readable English sentences describing the scene.
"""

import time


class SceneDescriber:
    """
    Rule-based NLG that generates scene descriptions.
    
    Takes all module outputs and produces sentences like:
    "1 person with a happy expression making a thumbs up gesture,
     near a laptop and coffee cup"
    """

    def __init__(self, update_interval=2.0):
        self.update_interval = update_interval
        self._last_update = 0
        self._cached_description = ""
        self._last_data = {}

    def update(self, object_detections=None, emotion_detections=None,
               gesture_detections=None, ocr_text="", relationships=None,
               activity=None):
        """
        Generate scene description from all module outputs.
        Only updates every `update_interval` seconds.

        Returns:
            The current scene description string.
        """
        now = time.time()
        if (now - self._last_update) < self.update_interval:
            return self._cached_description

        self._last_update = now

        parts = []

        # ── People and their states ──
        people_count = 0
        emotions = []
        gestures = []

        if object_detections:
            objects_by_type = {}
            for label, conf, bbox in object_detections:
                label_lower = label.lower()
                if label_lower == "person":
                    people_count += 1
                else:
                    objects_by_type[label_lower] = objects_by_type.get(label_lower, 0) + 1

            # Person description
            if people_count > 0:
                person_text = f"{people_count} person" if people_count == 1 else f"{people_count} people"
                person_parts = [person_text]

                # Add emotion
                if emotion_detections:
                    for emo, conf, _ in emotion_detections[:1]:
                        if conf > 0.4:
                            person_parts.append(f"with {emo} expression")
                            emotions.append(emo)

                # Add gesture
                if gesture_detections:
                    for gest, conf, _ in gesture_detections[:1]:
                        if conf > 0.5:
                            name = gest.split("|")[0].strip()
                            person_parts.append(f"making {name}")
                            gestures.append(name)

                # Add activity
                if activity:
                    person_parts.append(f"({activity})")

                parts.append(" ".join(person_parts))

            # Objects in scene
            if objects_by_type:
                obj_strs = []
                for name, count in list(objects_by_type.items())[:4]:
                    if count > 1:
                        obj_strs.append(f"{count} {name}s")
                    else:
                        obj_strs.append(f"a {name}")
                if obj_strs:
                    parts.append("near " + ", ".join(obj_strs))

        else:
            # No object detections — check other modules
            if emotion_detections:
                for emo, conf, _ in emotion_detections[:1]:
                    if conf > 0.4:
                        parts.append(f"A person with {emo} expression")
            if gesture_detections:
                for gest, conf, _ in gesture_detections[:1]:
                    name = gest.split("|")[0].strip()
                    parts.append(f"making {name}")

        # ── Relationships ──
        if relationships:
            rel_strs = [r for r in relationships[:2]]
            if rel_strs:
                parts.append("(" + "; ".join(rel_strs) + ")")

        # ── OCR ──
        if ocr_text and len(ocr_text.strip()) > 2:
            short = ocr_text.strip()[:40]
            parts.append(f'[Text visible: "{short}"]')

        # Build final description
        if parts:
            self._cached_description = " ".join(parts)
        else:
            self._cached_description = "Scene: No detections"

        return self._cached_description

    def get_description(self):
        return self._cached_description

    def draw(self, frame):
        """Draw scene description at the bottom of the frame."""
        import cv2

        if not self._cached_description:
            return

        h, w = frame.shape[:2]
        text = self._cached_description

        # Background bar at bottom
        bar_h = 35
        bar_y = h - bar_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (w, h), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # AI icon
        cv2.putText(
            frame, "AI:",
            (10, bar_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (100, 200, 255), 2, cv2.LINE_AA,
        )

        # Description text (truncate to fit)
        max_chars = int(w / 8)
        display_text = text[:max_chars]
        cv2.putText(
            frame, display_text,
            (45, bar_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (220, 220, 255), 1, cv2.LINE_AA,
        )
