"""
Hand Gesture Recognition Module using MediaPipe Tasks API.

Uses MediaPipe HandLandmarker (Tasks API) for 21-landmark hand
tracking with an improved rule-based classifier that uses MCP
joints as reference for more accurate finger state detection.
"""

import cv2
import math
import os
import config
from utils.one_euro_filter import HandLandmarkFilter


class HandGestureRecognizer:
    """
    MediaPipe-based hand gesture recognizer (Tasks API).
    
    Uses MCP joints as the reference baseline for finger extension
    detection, which is more reliable than PIP-only comparison.
    
    Counting:  0-5 fingers
    Gestures:  Thumbs Up/Down, Peace, Open Palm, Fist, Pointing Up,
               OK Sign, Rock/Metal, Call Me, I Love You, Pinch
    """

    # ── MediaPipe hand landmark indices ──
    WRIST = 0
    THUMB_CMC = 1; THUMB_MCP = 2; THUMB_IP = 3; THUMB_TIP = 4
    INDEX_MCP = 5; INDEX_PIP = 6; INDEX_DIP = 7; INDEX_TIP = 8
    MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
    RING_MCP = 13; RING_PIP = 14; RING_DIP = 15; RING_TIP = 16
    PINKY_MCP = 17; PINKY_PIP = 18; PINKY_DIP = 19; PINKY_TIP = 20

    GESTURE_NAMES = {
        "count_0": "Count: 0 (Fist)",
        "count_1": "Count: 1",
        "count_2": "Count: 2",
        "count_3": "Count: 3",
        "count_4": "Count: 4",
        "count_5": "Count: 5",
        "thumbs_up": "Thumbs Up",
        "thumbs_down": "Thumbs Down",
        "peace": "Peace / Victory",
        "open_palm": "Open Palm",
        "fist": "Fist",
        "pointing_up": "Pointing Up",
        "ok_sign": "OK Sign",
        "rock": "Rock / Metal",
        "call_me": "Call Me",
        "i_love_you": "I Love You",
        "pinch": "Pinch",
        "unknown": "Unknown Gesture",
    }

    def __init__(self, confidence_threshold=None):
        self.confidence_threshold = confidence_threshold or config.HAND_CONFIDENCE_THRESHOLD
        self.landmarker = None
        self._loaded = False
        self._last_landmarks = None  # Store smoothed landmarks for gesture control

        # One Euro Filter for landmark smoothing
        # min_cutoff=1.0: jitter suppression when still
        # beta=0.007: fast response when moving intentionally
        self._lm_filter = HandLandmarkFilter(min_cutoff=1.0, beta=0.007)

    def load_model(self):
        """Initialize MediaPipe HandLandmarker (Tasks API)."""
        print("[HandGesture] Loading MediaPipe HandLandmarker...")
        try:
            import mediapipe as mp
            from mediapipe.tasks.python.vision import (
                HandLandmarker,
                HandLandmarkerOptions,
                RunningMode,
            )
            from mediapipe.tasks.python import BaseOptions

            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models", "hand_landmarker.task"
            )

            if not os.path.exists(model_path):
                print("[HandGesture] Model file not found. Downloading...")
                import urllib.request
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                    model_path,
                )
                print("[HandGesture] Model downloaded.")

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.3,
            )

            self.landmarker = HandLandmarker.create_from_options(options)
            self.mp = mp
            self._loaded = True
            print("[HandGesture] MediaPipe HandLandmarker loaded successfully.")
        except Exception as e:
            print(f"[HandGesture] Error loading: {e}")
            import traceback
            traceback.print_exc()
            self._loaded = False

    # ──────────────────────────────────────────
    # Improved Finger State Detection
    # ──────────────────────────────────────────

    def _is_finger_extended(self, landmarks, tip_idx, dip_idx, pip_idx, mcp_idx):
        """
        Improved finger extension check using both DIP and MCP as references.
        
        A finger is extended if:
          - TIP is above PIP (basic check), AND
          - TIP is above MCP (stronger confirmation)
        
        This dual check reduces false positives from partially bent fingers.
        """
        tip_y = landmarks[tip_idx].y
        pip_y = landmarks[pip_idx].y
        mcp_y = landmarks[mcp_idx].y
        
        # Tip must be above both PIP and MCP (lower y = higher position)
        return tip_y < pip_y and tip_y < mcp_y

    def _is_finger_curled(self, landmarks, tip_idx, dip_idx, pip_idx, mcp_idx):
        """
        Check if a finger is definitely curled (closed).
        A finger is curled if tip is below PIP.
        """
        return landmarks[tip_idx].y > landmarks[pip_idx].y

    def _is_thumb_extended(self, landmarks, handedness):
        """
        Check if thumb is extended outward based on horizontal distance.
        Uses both tip-to-ip and tip-to-mcp distances for reliability.
        """
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        thumb_mcp = landmarks[self.THUMB_MCP]
        wrist = landmarks[self.WRIST]
        index_mcp = landmarks[self.INDEX_MCP]

        # Calculate how far the thumb tip is from the palm center
        palm_center_x = (wrist.x + index_mcp.x) / 2

        if handedness == "Right":
            # Right hand: thumb extends to the left (lower x)
            return thumb_tip.x < thumb_mcp.x and abs(thumb_tip.x - palm_center_x) > 0.05
        else:
            # Left hand: thumb extends to the right (higher x)
            return thumb_tip.x > thumb_mcp.x and abs(thumb_tip.x - palm_center_x) > 0.05

    def _is_thumb_up(self, landmarks):
        """Check if thumb is pointing upward relative to wrist."""
        thumb_tip = landmarks[self.THUMB_TIP]
        wrist = landmarks[self.WRIST]
        index_mcp = landmarks[self.INDEX_MCP]
        # Thumb tip should be significantly above wrist and index MCP
        return thumb_tip.y < wrist.y - 0.1 and thumb_tip.y < index_mcp.y

    def _is_thumb_down(self, landmarks):
        """Check if thumb is pointing downward relative to wrist."""
        thumb_tip = landmarks[self.THUMB_TIP]
        wrist = landmarks[self.WRIST]
        # Thumb tip should be significantly below wrist
        return thumb_tip.y > wrist.y + 0.05

    def _get_finger_states(self, landmarks, handedness):
        """Get the extension state of all 5 fingers with improved detection."""
        thumb = self._is_thumb_extended(landmarks, handedness)
        index = self._is_finger_extended(
            landmarks, self.INDEX_TIP, self.INDEX_DIP, self.INDEX_PIP, self.INDEX_MCP
        )
        middle = self._is_finger_extended(
            landmarks, self.MIDDLE_TIP, self.MIDDLE_DIP, self.MIDDLE_PIP, self.MIDDLE_MCP
        )
        ring = self._is_finger_extended(
            landmarks, self.RING_TIP, self.RING_DIP, self.RING_PIP, self.RING_MCP
        )
        pinky = self._is_finger_extended(
            landmarks, self.PINKY_TIP, self.PINKY_DIP, self.PINKY_PIP, self.PINKY_MCP
        )
        return [thumb, index, middle, ring, pinky]

    def _distance(self, lm1, lm2):
        """Euclidean distance between two landmarks."""
        return math.sqrt((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2)

    def _are_tips_touching(self, landmarks, tip1_idx, tip2_idx, threshold=0.06):
        """Check if two fingertips are close together."""
        return self._distance(landmarks[tip1_idx], landmarks[tip2_idx]) < threshold

    # ──────────────────────────────────────────
    # Gesture Classification
    # ──────────────────────────────────────────

    def _classify_gesture(self, landmarks, handedness):
        """
        Classify a hand gesture with improved accuracy.

        Returns:
            Tuple of (gesture_key, confidence, finger_count).
        """
        fingers = self._get_finger_states(landmarks, handedness)
        thumb, index, middle, ring, pinky = fingers
        finger_count = sum(fingers)

        # Also check if fingers are definitely curled
        index_curled = self._is_finger_curled(
            landmarks, self.INDEX_TIP, self.INDEX_DIP, self.INDEX_PIP, self.INDEX_MCP
        )
        middle_curled = self._is_finger_curled(
            landmarks, self.MIDDLE_TIP, self.MIDDLE_DIP, self.MIDDLE_PIP, self.MIDDLE_MCP
        )
        ring_curled = self._is_finger_curled(
            landmarks, self.RING_TIP, self.RING_DIP, self.RING_PIP, self.RING_MCP
        )
        pinky_curled = self._is_finger_curled(
            landmarks, self.PINKY_TIP, self.PINKY_DIP, self.PINKY_PIP, self.PINKY_MCP
        )

        # ── Special gestures (check first) ──

        # OK Sign: thumb + index tips touching, others extended
        if self._are_tips_touching(landmarks, self.THUMB_TIP, self.INDEX_TIP):
            if middle and ring and pinky:
                return "ok_sign", 0.92, 3  # 3 fingers clearly out

        # Pinch: thumb + index touching, others curled
        if self._are_tips_touching(landmarks, self.THUMB_TIP, self.INDEX_TIP):
            if middle_curled and ring_curled and pinky_curled:
                return "pinch", 0.88, 0

        # Thumbs Up: only thumb extended AND pointing up
        if thumb and index_curled and middle_curled and ring_curled and pinky_curled:
            if self._is_thumb_up(landmarks):
                return "thumbs_up", 0.92, 1

        # Thumbs Down: only thumb extended AND pointing down
        if thumb and index_curled and middle_curled and ring_curled and pinky_curled:
            if self._is_thumb_down(landmarks):
                return "thumbs_down", 0.90, 1

        # Fist: all fingers curled (check explicitly)
        if index_curled and middle_curled and ring_curled and pinky_curled and not thumb:
            return "fist", 0.90, 0

        # Rock / Metal: index + pinky up, middle + ring curled
        if index and pinky and middle_curled and ring_curled and not thumb:
            return "rock", 0.90, 2

        # I Love You: thumb + index + pinky up, middle + ring curled
        if thumb and index and pinky and middle_curled and ring_curled:
            return "i_love_you", 0.88, 3

        # Call Me: thumb + pinky up, others curled
        if thumb and pinky and index_curled and middle_curled and ring_curled:
            return "call_me", 0.88, 2

        # Peace / Victory: index + middle up, others curled
        if index and middle and ring_curled and pinky_curled and not thumb:
            return "peace", 0.90, 2

        # Pointing Up: only index extended
        if index and not thumb and middle_curled and ring_curled and pinky_curled:
            return "pointing_up", 0.90, 1

        # Open Palm: all fingers extended
        if finger_count == 5:
            return "open_palm", 0.92, 5

        # ── Counting fallback ──
        if finger_count == 0:
            return "count_0", 0.85, 0
        elif finger_count == 1:
            return "count_1", 0.80, 1
        elif finger_count == 2:
            return "count_2", 0.80, 2
        elif finger_count == 3:
            return "count_3", 0.80, 3
        elif finger_count == 4:
            return "count_4", 0.80, 4

        return "unknown", 0.40, finger_count

    def _draw_hand_skeleton(self, frame, landmarks, h, w):
        """Draw hand skeleton with styled landmarks and connections."""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        ]

        # Draw connections
        for start_idx, end_idx in connections:
            s = landmarks[start_idx]
            e = landmarks[end_idx]
            pt1 = (int(s.x * w), int(s.y * h))
            pt2 = (int(e.x * w), int(e.y * h))
            cv2.line(frame, pt1, pt2, (0, 255, 200), 2, cv2.LINE_AA)

        # Draw landmark points
        for i, lm in enumerate(landmarks):
            px, py = int(lm.x * w), int(lm.y * h)
            # Tips get larger circles
            radius = 5 if i in (4, 8, 12, 16, 20) else 3
            cv2.circle(frame, (px, py), radius, (255, 0, 100), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), radius + 2, (255, 255, 255), 1, cv2.LINE_AA)



    def detect(self, frame):
        """
        Detect hands and classify gestures.

        Args:
            frame: BGR image (numpy array from OpenCV).

        Returns:
            List of tuples: (gesture_display_name, confidence, (x1, y1, x2, y2))
        """
        if not self._loaded:
            return []

        try:
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = self.mp.Image(
                image_format=self.mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            result = self.landmarker.detect(mp_image)

            detections = []
            if result.hand_landmarks:
                for i, hand_landmarks in enumerate(result.hand_landmarks):
                    landmarks = hand_landmarks

                    # ── One Euro Filter: low jitter when still, fast when moving ──
                    if i == 0:
                        landmarks = self._lm_filter.apply(landmarks)

                    # Store smoothed landmarks for gesture control module
                    if i == 0:
                        self._last_landmarks = landmarks

                    handedness = "Right"
                    if result.handedness and i < len(result.handedness):
                        handedness = result.handedness[i][0].category_name

                    gesture_key, confidence, finger_count = self._classify_gesture(
                        landmarks, handedness
                    )
                    gesture_name = self.GESTURE_NAMES.get(gesture_key, "Unknown")
                    display_text = f"{gesture_name} | Fingers: {finger_count}"

                    x_coords = [lm.x * w for lm in landmarks]
                    y_coords = [lm.y * h for lm in landmarks]

                    margin = 25
                    x1 = max(0, int(min(x_coords)) - margin)
                    y1 = max(0, int(min(y_coords)) - margin)
                    x2 = min(w, int(max(x_coords)) + margin)
                    y2 = min(h, int(max(y_coords)) + margin)

                    bbox = (x1, y1, x2, y2)
                    detections.append((display_text, confidence, bbox))

                    self._draw_hand_skeleton(frame, landmarks, h, w)

            else:
                # Hand left frame — reset filter and clear landmarks
                self._last_landmarks = None
                self._lm_filter.reset()

            return detections

        except Exception as e:
            print(f"[HandGesture] Detection error: {e}")
            self._last_landmarks = None
            return []

    def is_loaded(self):
        return self._loaded

    def release(self):
        if self.landmarker:
            self.landmarker.close()
