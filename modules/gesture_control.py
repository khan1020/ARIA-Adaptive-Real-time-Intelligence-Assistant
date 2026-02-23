"""
Gesture-Controlled Computer Module.

Uses hand landmarks from MediaPipe to control the computer:
  - Index finger moves the mouse cursor (virtual mouse)
  - Pinch (thumb + index close) = Left Click
  - Fist (held 1s) = Mute / Unmute
  - Thumbs Up = Play / Pause media
  - Peace sign (held 2s) = Screenshot
  - Open Palm + move up/down = Scroll

Uses pyautogui for mouse/keyboard and pycaw for volume (with fallback).
"""

import cv2
import time
import math
import os


class GestureController:
    """
    Maps hand gestures and finger positions to system actions.

    Virtual Mouse: Index finger tip position maps to screen coordinates.
    The camera frame is divided into a control zone. When the index
    finger moves within this zone, the mouse cursor follows.
    """

    # Actions
    ACTION_NONE = ""
    ACTION_CLICK = "Left Click"
    ACTION_PLAY_PAUSE = "Play/Pause"
    ACTION_MUTE = "Mute/Unmute"
    ACTION_SCROLL_UP = "Scroll Up"
    ACTION_SCROLL_DOWN = "Scroll Down"
    ACTION_SCREENSHOT = "Screenshot"
    ACTION_VOL_UP = "Volume Up"
    ACTION_VOL_DOWN = "Volume Down"

    def __init__(self):
        self._loaded = False
        self._volume = None
        self._pyautogui = None
        self._screen_w = 1920
        self._screen_h = 1080

        # Mouse smoothing
        self._prev_mouse_x = 0
        self._prev_mouse_y = 0
        self._smoothing = 0.25  # Lower = smoother, higher = more responsive

        # Click detection
        self._pinch_was_close = False
        self._pinch_click_cooldown = 0

        # Gesture hold timers
        self._gesture_start_time = {}
        self._action_cooldown = {}
        self._cooldown_seconds = 1.0
        self._last_action = ""
        self._last_action_time = 0

        # Scroll tracking
        self._prev_palm_y = None
        self._scroll_threshold = 0.03

        # Mouse control active
        self._mouse_active = False

    def load(self):
        """Initialize system control libraries."""
        print("[GestureControl] Initializing system control...")
        try:
            import pyautogui
            pyautogui.FAILSAFE = True   # Move mouse to any screen CORNER to abort gesture control
            pyautogui.PAUSE = 0         # No delay between actions (needed for smooth mouse)
            self._pyautogui = pyautogui

            # Get actual screen size
            self._screen_w, self._screen_h = pyautogui.size()
            print(f"[GestureControl] Screen: {self._screen_w}x{self._screen_h}")

            # Windows volume control via pycaw
            try:
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL

                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(
                    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
                )
                self._volume = cast(interface, POINTER(IAudioEndpointVolume))
                print("[GestureControl] Volume control ready.")
            except Exception:
                print("[GestureControl] pycaw unavailable, using keyboard fallback.")
                self._volume = None

            self._loaded = True
            print("[GestureControl] System control ready.")
        except Exception as e:
            print(f"[GestureControl] Init failed: {e}")
            self._loaded = False

    def _can_trigger(self, action):
        now = time.time()
        last = self._action_cooldown.get(action, 0)
        return (now - last) >= self._cooldown_seconds

    def _trigger(self, action):
        self._action_cooldown[action] = time.time()
        self._last_action = action
        self._last_action_time = time.time()

    def _toggle_mute(self):
        if self._volume:
            try:
                muted = self._volume.GetMute()
                self._volume.SetMute(not muted, None)
                return
            except Exception:
                pass
        if self._pyautogui:
            self._pyautogui.press('volumemute')

    def process(self, gesture_detections, hand_landmarks=None):
        """
        Process hand data and control the computer.

        Args:
            gesture_detections: List of (gesture_name, conf, bbox) from hand module
            hand_landmarks: Raw MediaPipe hand landmarks (21 points)

        Returns:
            Action string that was triggered (or empty).
        """
        if not self._loaded:
            return ""

        # No hand detected — reset state
        if not gesture_detections or not hand_landmarks:
            self._prev_palm_y = None
            self._pinch_was_close = False
            self._mouse_active = False
            return ""

        gesture_text = gesture_detections[0][0].lower()
        now = time.time()

        # Get key landmark positions (normalized 0-1)
        index_tip = hand_landmarks[8]   # Index finger tip
        thumb_tip = hand_landmarks[4]   # Thumb tip
        middle_tip = hand_landmarks[12] # Middle finger tip
        wrist = hand_landmarks[0]

        # ══════════════════════════════════════
        # 1. VIRTUAL MOUSE — Index finger up only
        # ══════════════════════════════════════
        # When only index finger is pointing (Count: 1 or Pointing Up)
        if "count: 1" in gesture_text or "pointing" in gesture_text:
            self._mouse_active = True
            self._move_mouse(index_tip.x, index_tip.y)

            # Check for pinch click — distance between thumb and index
            # Not applicable in pointing mode
            self._prev_palm_y = None
            return ""

        # ══════════════════════════════════════
        # 2. PINCH CLICK — Thumb and index together (Count: 2 or Pinch)
        # ══════════════════════════════════════
        if "pinch" in gesture_text or "count: 2" in gesture_text or "ok" in gesture_text:
            # Use index tip position for mouse
            self._move_mouse(index_tip.x, index_tip.y)

            thumb_index_dist = math.sqrt(
                (thumb_tip.x - index_tip.x) ** 2 +
                (thumb_tip.y - index_tip.y) ** 2
            )

            # Pinch detected (thumb and index very close)
            if thumb_index_dist < 0.05:
                if not self._pinch_was_close and self._can_trigger(self.ACTION_CLICK):
                    self._pyautogui.click()
                    self._trigger(self.ACTION_CLICK)
                    self._pinch_was_close = True
                    self._prev_palm_y = None
                    return self.ACTION_CLICK
            else:
                self._pinch_was_close = False

            self._prev_palm_y = None
            return ""

        # ══════════════════════════════════════
        # 3. SCROLL — Open Palm, track vertical movement
        # ══════════════════════════════════════
        if "open palm" in gesture_text or "count: 5" in gesture_text:
            palm_y = wrist.y
            if self._prev_palm_y is not None:
                dy = palm_y - self._prev_palm_y
                if abs(dy) > self._scroll_threshold:
                    if dy < 0 and self._can_trigger(self.ACTION_SCROLL_UP):
                        self._pyautogui.scroll(5)
                        self._trigger(self.ACTION_SCROLL_UP)
                        self._prev_palm_y = palm_y
                        return self.ACTION_SCROLL_UP
                    elif dy > 0 and self._can_trigger(self.ACTION_SCROLL_DOWN):
                        self._pyautogui.scroll(-5)
                        self._trigger(self.ACTION_SCROLL_DOWN)
                        self._prev_palm_y = palm_y
                        return self.ACTION_SCROLL_DOWN
            self._prev_palm_y = palm_y
            self._mouse_active = False
            return ""
        else:
            self._prev_palm_y = None

        # ══════════════════════════════════════
        # 4. PLAY/PAUSE — Thumbs Up
        # ══════════════════════════════════════
        if "thumbs up" in gesture_text:
            if self._can_trigger(self.ACTION_PLAY_PAUSE):
                self._pyautogui.press('playpause')
                self._trigger(self.ACTION_PLAY_PAUSE)
                return self.ACTION_PLAY_PAUSE

        # ══════════════════════════════════════
        # 5. MUTE — Fist (hold 1 second)
        # ══════════════════════════════════════
        if "fist" in gesture_text or "count: 0" in gesture_text:
            if "fist" not in self._gesture_start_time:
                self._gesture_start_time["fist"] = now
            elif (now - self._gesture_start_time["fist"]) > 1.0:
                if self._can_trigger(self.ACTION_MUTE):
                    self._toggle_mute()
                    self._trigger(self.ACTION_MUTE)
                    self._gesture_start_time.pop("fist", None)
                    return self.ACTION_MUTE
        else:
            self._gesture_start_time.pop("fist", None)

        # ══════════════════════════════════════
        # 6. SCREENSHOT — Peace sign (hold 2 seconds)
        # ══════════════════════════════════════
        if "peace" in gesture_text or "victory" in gesture_text:
            if "peace" not in self._gesture_start_time:
                self._gesture_start_time["peace"] = now
            elif (now - self._gesture_start_time["peace"]) > 2.0:
                if self._can_trigger(self.ACTION_SCREENSHOT):
                    screenshot_dir = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "screenshots"
                    )
                    os.makedirs(screenshot_dir, exist_ok=True)
                    import datetime
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision
                    filename = os.path.join(screenshot_dir, f"screenshot_{ts}.png")
                    self._pyautogui.screenshot(filename)
                    self._trigger(self.ACTION_SCREENSHOT)
                    self._gesture_start_time.pop("peace", None)
                    return self.ACTION_SCREENSHOT
        else:
            self._gesture_start_time.pop("peace", None)

        self._mouse_active = False
        return ""

    def _move_mouse(self, norm_x, norm_y):
        """
        Move real mouse cursor based on normalized hand position.

        Maps the center 60% of the camera frame to full screen.
        Uses heavy smoothing + dead zone to reduce jitter.
        """
        # Map from camera space to screen space
        # Use center 60% of frame as the active zone for more precision
        margin = 0.2
        clamped_x = max(0, min(1, (norm_x - margin) / (1 - 2 * margin)))
        clamped_y = max(0, min(1, (norm_y - margin) / (1 - 2 * margin)))

        # Invert X because frame is flipped horizontally
        target_x = int((1 - clamped_x) * self._screen_w)
        target_y = int(clamped_y * self._screen_h)

        # Smooth movement (exponential moving average, lower = smoother)
        smooth_x = int(self._prev_mouse_x + self._smoothing * (target_x - self._prev_mouse_x))
        smooth_y = int(self._prev_mouse_y + self._smoothing * (target_y - self._prev_mouse_y))

        # Dead zone: ignore movement smaller than 5 pixels
        dx = abs(smooth_x - self._prev_mouse_x)
        dy = abs(smooth_y - self._prev_mouse_y)
        if dx < 5 and dy < 5:
            return

        self._prev_mouse_x = smooth_x
        self._prev_mouse_y = smooth_y

        try:
            self._pyautogui.moveTo(smooth_x, smooth_y, _pause=False)
        except Exception:
            pass

    def get_last_action(self):
        if time.time() - self._last_action_time < 2.0:
            return self._last_action
        return ""

    def draw(self, frame):
        """Draw gesture control status overlay."""
        h, w = frame.shape[:2]

        # ── Mouse active indicator ──
        if self._mouse_active:
            cv2.putText(frame, "MOUSE", (w // 2 - 35, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 200), 2, cv2.LINE_AA)
            # Draw crosshair at current mouse position mapped to frame
            cx = int((1 - self._prev_mouse_x / self._screen_w) * w)
            cy = int(self._prev_mouse_y / self._screen_h * h)
            cv2.drawMarker(frame, (cx, cy), (0, 255, 200),
                           cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

        # ── Action popup ──
        action = self.get_last_action()
        if action:
            text = f">> {action} <<"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            tx = (w - tw) // 2
            ty = 55 if not self._mouse_active else 65

            overlay = frame.copy()
            cv2.rectangle(overlay, (tx - 15, ty - 30), (tx + tw + 15, ty + 10), (0, 50, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 100), 2, cv2.LINE_AA)

        # ── Volume bar ──
        if self._volume:
            try:
                vol = self._volume.GetMasterVolumeLevelScalar()
                bar_x = w - 35
                bar_h_px = 120
                bar_y = 80
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 20, bar_y + bar_h_px), (40, 40, 40), -1)
                fill_h = int(vol * bar_h_px)
                cv2.rectangle(frame, (bar_x, bar_y + bar_h_px - fill_h), (bar_x + 20, bar_y + bar_h_px), (0, 255, 100), -1)
                cv2.putText(frame, f"{int(vol*100)}%", (bar_x - 5, bar_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
            except Exception:
                pass

    def is_loaded(self):
        return self._loaded
