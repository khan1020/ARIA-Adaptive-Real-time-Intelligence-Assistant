"""
Voice Feedback Module — Text-to-Speech for detections.

Uses pyttsx3 for offline TTS in a background thread.
Includes category-based cooldowns to avoid spamming.
Re-announces the same detection after cooldown expires.
"""

import threading
import time
import queue


class VoiceFeedback:
    """
    Background TTS engine that announces detections.

    Fixes:
    - Re-creates engine per utterance to avoid pyttsx3 thread deadlocks
    - Re-announces same text after cooldown (doesn't skip forever)
    - Uses queue.Queue for thread-safe communication
    """

    def __init__(self, cooldown_seconds=4):
        self.cooldown = cooldown_seconds
        self._enabled = True
        self._loaded = False
        self._last_spoken_time = {}  # category -> timestamp
        self._queue = queue.Queue(maxsize=5)
        self._thread = None
        self._running = False

    def load(self):
        """Initialize the TTS system."""
        print("[VoiceFeedback] Initializing TTS engine...")
        try:
            import pyttsx3
            # Test that engine can initialize
            engine = pyttsx3.init()
            engine.stop()
            del engine
            self._loaded = True
            self._running = True
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
            print("[VoiceFeedback] TTS engine ready.")
        except Exception as e:
            print(f"[VoiceFeedback] TTS init failed: {e}")
            self._loaded = False

    def _worker(self):
        """Background thread — creates fresh engine per utterance."""
        import pyttsx3

        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 155)
            engine.setProperty('volume', 0.9)
        except Exception:
            self._loaded = False
            return

        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
                if text is None:
                    continue
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    # Re-create engine if it crashes
                    try:
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 155)
                        engine.setProperty('volume', 0.9)
                    except Exception:
                        pass
            except queue.Empty:
                continue

    def announce(self, category, text):
        """
        Queue an announcement if cooldown has elapsed.
        Will re-announce the same text after cooldown expires.
        """
        if not self._loaded or not self._enabled:
            return

        now = time.time()
        last_time = self._last_spoken_time.get(category, 0)

        # Check cooldown — if still within cooldown, skip
        if (now - last_time) < self.cooldown:
            return

        self._last_spoken_time[category] = now

        # Try to add to queue (non-blocking)
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass  # Queue full, skip this one

    def announce_objects(self, detections):
        """Announce detected objects."""
        if not detections:
            return
        names = list(set(label for label, _, _ in detections))[:3]
        if names:
            self.announce("objects", f"I see {', '.join(names)}")

    def announce_emotion(self, detections):
        """Announce detected emotions."""
        if not detections:
            return
        for emotion, conf, _ in detections:
            if conf > 0.35:
                self.announce("emotion", f"{emotion} expression detected")
                break

    def announce_gesture(self, detections):
        """Announce detected gestures."""
        if not detections:
            return
        for gesture, conf, _ in detections:
            if conf > 0.5:
                name = gesture.split("|")[0].strip()
                self.announce("gesture", name)
                break

    def announce_ocr(self, text):
        """Announce OCR result."""
        if text and len(text.strip()) > 2:
            short = text.strip()[:50]
            self.announce("ocr", f"Text reads: {short}")

    def announce_activity(self, activity):
        """Announce current activity."""
        if activity and activity.lower() != "idle":
            self.announce("activity", f"Activity: {activity}")

    def announce_relations(self, relations):
        """Announce object relationships."""
        if relations:
            self.announce("relation", relations[0])

    def toggle(self):
        """Toggle voice on/off."""
        self._enabled = not self._enabled
        return self._enabled

    def is_enabled(self):
        return self._enabled

    def is_loaded(self):
        return self._loaded

    def shutdown(self):
        self._running = False
