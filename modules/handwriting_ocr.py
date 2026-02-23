"""
Handwritten Text Recognition Module with Hand-Drawn Canvas.

Provides a drawable canvas where the user draws using their
INDEX FINGER tracked by MediaPipe. Gestures control tools:
  - Point (index only) = Draw
  - Open Palm = Hover/Pause
  - Fist = Erase
  - Peace Sign = Clear canvas

Mouse is available as fallback. Canvas content is processed
by Tesseract OCR to recognize handwritten text.
"""

import cv2
import numpy as np
import math
import os
import config


class HandwritingOCR:
    """
    Tesseract OCR with hand-tracked drawing canvas.
    
    Uses MediaPipe HandLandmarker to track the index finger tip
    as a virtual pen. Different hand gestures switch between
    drawing, erasing, and clearing.
    """

    # Landmark indices
    WRIST = 0
    THUMB_TIP = 4; THUMB_IP = 3; THUMB_MCP = 2
    INDEX_TIP = 8; INDEX_PIP = 6; INDEX_MCP = 5; INDEX_DIP = 7
    MIDDLE_TIP = 12; MIDDLE_PIP = 10; MIDDLE_MCP = 9
    RING_TIP = 16; RING_PIP = 14
    PINKY_TIP = 20; PINKY_PIP = 18

    # Tool modes
    MODE_DRAW = "DRAW"
    MODE_ERASE = "ERASE"
    MODE_HOVER = "HOVER"
    MODE_CLEAR = "CLEAR"

    # Canvas UI
    CANVAS_WIDTH = 460
    CANVAS_HEIGHT = 320

    def __init__(self):
        self.pytesseract = None
        self._loaded = False
        self._hand_loaded = False

        # Hand tracking
        self.landmarker = None
        self.mp = None

        # Canvas
        self.canvas = None
        self.canvas_x = 0
        self.canvas_y = 0
        self._reset_canvas()

        # Drawing state
        self.current_mode = self.MODE_HOVER
        self.last_draw_point = None
        self.pencil_size = 3
        self.eraser_size = 25

        # Smoothing — weighted moving average over N frames
        self._finger_history = []
        self._history_len = 20

        # Dead zone — minimum pixel movement required to draw
        self._min_move_px = 8

        # OCR
        self._frame_count = 0
        self._process_every_n = 25
        self._cached_text = ""

        # Finger cursor position for display
        self._cursor_pos = None

    def _reset_canvas(self):
        """Clear the canvas to white."""
        self.canvas = np.ones(
            (self.CANVAS_HEIGHT, self.CANVAS_WIDTH, 3), dtype=np.uint8
        ) * 255
        self._cached_text = ""
        self.last_draw_point = None

    def load_model(self):
        """Load Tesseract OCR and MediaPipe HandLandmarker."""
        print("[HandwritingOCR] Loading Tesseract OCR...")
        try:
            import pytesseract

            if os.path.exists(config.TESSERACT_PATH):
                pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_PATH

            version = pytesseract.get_tesseract_version()
            self.pytesseract = pytesseract
            self._loaded = True
            print(f"[HandwritingOCR] Tesseract v{version} loaded.")
        except Exception as e:
            print(f"[HandwritingOCR] Error loading Tesseract: {e}")
            self._loaded = False

        # Load MediaPipe for hand tracking
        print("[HandwritingOCR] Loading hand tracker for drawing...")
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
                print("[HandwritingOCR] Downloading hand model...")
                import urllib.request
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                    model_path,
                )

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_hands=1,  # Single hand for drawing
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.4,
            )

            self.landmarker = HandLandmarker.create_from_options(options)
            self.mp = mp
            self._hand_loaded = True
            print("[HandwritingOCR] Hand tracker loaded.")
        except Exception as e:
            print(f"[HandwritingOCR] Hand tracker failed: {e}")
            self._hand_loaded = False

    # ──────────────────────────────────────────
    # Finger Detection
    # ──────────────────────────────────────────

    def _finger_extended(self, lms, tip, pip):
        return lms[tip].y < lms[pip].y

    def _finger_curled(self, lms, tip, pip):
        return lms[tip].y > lms[pip].y

    def _detect_draw_gesture(self, landmarks, handedness):
        """
        Detect drawing gesture from hand landmarks.
        
        Returns:
            (mode, finger_tip_x, finger_tip_y) in normalized coords
        """
        lms = landmarks

        index_up = self._finger_extended(lms, self.INDEX_TIP, self.INDEX_PIP)
        middle_up = self._finger_extended(lms, self.MIDDLE_TIP, self.MIDDLE_PIP)
        ring_up = self._finger_extended(lms, self.RING_TIP, self.RING_PIP)
        pinky_up = self._finger_extended(lms, self.PINKY_TIP, self.PINKY_PIP)

        index_curled = self._finger_curled(lms, self.INDEX_TIP, self.INDEX_PIP)
        middle_curled = self._finger_curled(lms, self.MIDDLE_TIP, self.MIDDLE_PIP)
        ring_curled = self._finger_curled(lms, self.RING_TIP, self.RING_PIP)
        pinky_curled = self._finger_curled(lms, self.PINKY_TIP, self.PINKY_PIP)

        finger_count = sum([index_up, middle_up, ring_up, pinky_up])

        # Get finger tip position
        tip_x = lms[self.INDEX_TIP].x
        tip_y = lms[self.INDEX_TIP].y

        # DRAW: Only index finger raised (pointing)
        if index_up and middle_curled and ring_curled and pinky_curled:
            return self.MODE_DRAW, tip_x, tip_y

        # ERASE: Fist (all fingers curled)
        if index_curled and middle_curled and ring_curled and pinky_curled:
            return self.MODE_ERASE, tip_x, tip_y

        # CLEAR: Peace/Victory sign (index + middle up)
        if index_up and middle_up and ring_curled and pinky_curled:
            return self.MODE_CLEAR, tip_x, tip_y

        # HOVER: Open palm or any other gesture
        return self.MODE_HOVER, tip_x, tip_y

    def _smooth_position(self, x, y):
        """Apply weighted moving average — recent positions matter more."""
        self._finger_history.append((x, y))
        if len(self._finger_history) > self._history_len:
            self._finger_history.pop(0)

        n = len(self._finger_history)
        # Weights: 1, 2, 3, ... n  (newer = heavier)
        total_w = 0
        avg_x = 0.0
        avg_y = 0.0
        for i, (px, py) in enumerate(self._finger_history):
            w = i + 1
            avg_x += px * w
            avg_y += py * w
            total_w += w
        return avg_x / total_w, avg_y / total_w

    def _map_to_canvas(self, norm_x, norm_y, frame_w, frame_h):
        """
        Map normalized finger coordinates to canvas pixel coordinates.
        Returns None if finger is outside the canvas area.
        """
        # Convert normalized to frame pixels
        fx = int(norm_x * frame_w)
        fy = int(norm_y * frame_h)

        # Check if finger is within the canvas bounds
        cx = fx - self.canvas_x
        cy = fy - self.canvas_y

        if 0 <= cx < self.CANVAS_WIDTH and 0 <= cy < self.CANVAS_HEIGHT:
            return cx, cy
        return None

    # ──────────────────────────────────────────
    # Mouse Fallback
    # ──────────────────────────────────────────

    def handle_mouse(self, event, x, y, flags, param):
        """Mouse-based drawing as fallback."""
        cx = x - self.canvas_x
        cy = y - self.canvas_y

        if 0 <= cx < self.CANVAS_WIDTH and 0 <= cy < self.CANVAS_HEIGHT:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._mouse_drawing = True
                self._mouse_last = (cx, cy)
                cv2.circle(self.canvas, (cx, cy), self.pencil_size, (0, 0, 0), -1, cv2.LINE_AA)
            elif event == cv2.EVENT_MOUSEMOVE and getattr(self, '_mouse_drawing', False):
                if self._mouse_last:
                    cv2.line(self.canvas, self._mouse_last, (cx, cy), (0, 0, 0), self.pencil_size * 2, cv2.LINE_AA)
                self._mouse_last = (cx, cy)
            elif event == cv2.EVENT_LBUTTONUP:
                self._mouse_drawing = False
                self._mouse_last = None

            # Right-click to erase
            if event == cv2.EVENT_RBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_RBUTTON):
                cv2.circle(self.canvas, (cx, cy), self.eraser_size, (255, 255, 255), -1)
        else:
            if event == cv2.EVENT_LBUTTONUP:
                self._mouse_drawing = False

    # ──────────────────────────────────────────
    # Hand-Based Drawing
    # ──────────────────────────────────────────

    def _process_hand(self, frame):
        """
        Track hand and draw on the canvas based on gestures.
        
        Args:
            frame: The video frame (already flipped).
        """
        if not self._hand_loaded:
            return

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect(mp_image)

        if not result.hand_landmarks:
            self.last_draw_point = None
            self._cursor_pos = None
            self._finger_history.clear()
            if self.current_mode != self.MODE_HOVER:
                self.current_mode = self.MODE_HOVER
            return

        landmarks = result.hand_landmarks[0]
        handedness = "Right"
        if result.handedness and len(result.handedness) > 0:
            handedness = result.handedness[0][0].category_name

        # Detect gesture and get finger position
        mode, tip_x, tip_y = self._detect_draw_gesture(landmarks, handedness)

        # Smooth the position
        tip_x, tip_y = self._smooth_position(tip_x, tip_y)

        # Screen-space cursor position for visual feedback
        self._cursor_pos = (int(tip_x * w), int(tip_y * h))

        # Map to canvas coordinates
        canvas_pt = self._map_to_canvas(tip_x, tip_y, w, h)

        self.current_mode = mode

        if mode == self.MODE_CLEAR:
            self._reset_canvas()
            return

        if canvas_pt is None:
            # Finger is outside canvas
            self.last_draw_point = None
            return

        cx, cy = canvas_pt

        if mode == self.MODE_DRAW:
            if self.last_draw_point is not None:
                # Dead zone: only draw if finger moved enough pixels
                dx = cx - self.last_draw_point[0]
                dy = cy - self.last_draw_point[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist >= self._min_move_px:
                    cv2.line(
                        self.canvas, self.last_draw_point, (cx, cy),
                        (0, 0, 0), self.pencil_size * 2, cv2.LINE_AA,
                    )
                    self.last_draw_point = (cx, cy)
            else:
                self.last_draw_point = (cx, cy)

        elif mode == self.MODE_ERASE:
            cv2.circle(
                self.canvas, (cx, cy), self.eraser_size,
                (255, 255, 255), -1,
            )
            self.last_draw_point = None

        elif mode == self.MODE_HOVER:
            self.last_draw_point = None

    # ──────────────────────────────────────────
    # Canvas Drawing & UI
    # ──────────────────────────────────────────

    def _draw_ui(self, frame):
        """Draw the canvas, gesture guide, and OCR results onto the frame."""
        h, w = frame.shape[:2]

        # Position canvas on the right
        self.canvas_x = w - self.CANVAS_WIDTH - 15
        self.canvas_y = 70

        # ── Gesture guide bar ──
        guide_y = self.canvas_y - 25
        guides = [
            ("Point=DRAW", (0, 255, 0)),
            ("Fist=ERASE", (0, 100, 255)),
            ("Peace=CLEAR", (255, 0, 255)),
            ("Palm=PAUSE", (200, 200, 0)),
        ]
        gx = self.canvas_x
        for text, color in guides:
            cv2.putText(frame, text, (gx, guide_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            gx += 115

        # ── Current mode indicator ──
        mode_colors = {
            self.MODE_DRAW: (0, 255, 0),
            self.MODE_ERASE: (0, 100, 255),
            self.MODE_HOVER: (200, 200, 200),
            self.MODE_CLEAR: (255, 0, 255),
        }
        mode_color = mode_colors.get(self.current_mode, (200, 200, 200))
        cv2.putText(
            frame, f"Mode: {self.current_mode}",
            (self.canvas_x, self.canvas_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 2, cv2.LINE_AA,
        )

        # Also show mouse hint
        cv2.putText(
            frame, "(Mouse: L=draw, R=erase)",
            (self.canvas_x + 180, self.canvas_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA,
        )

        # ── Canvas border ──
        border_color = mode_color
        cv2.rectangle(
            frame,
            (self.canvas_x - 3, self.canvas_y - 3),
            (self.canvas_x + self.CANVAS_WIDTH + 3, self.canvas_y + self.CANVAS_HEIGHT + 3),
            border_color, 2,
        )

        # ── Overlay canvas ──
        frame[
            self.canvas_y: self.canvas_y + self.CANVAS_HEIGHT,
            self.canvas_x: self.canvas_x + self.CANVAS_WIDTH,
        ] = self.canvas

        # ── Draw cursor indicator ──
        if self._cursor_pos:
            cx, cy = self._cursor_pos
            if self.current_mode == self.MODE_DRAW:
                cv2.circle(frame, (cx, cy), self.pencil_size + 3, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1, cv2.LINE_AA)
            elif self.current_mode == self.MODE_ERASE:
                cv2.circle(frame, (cx, cy), self.eraser_size, (0, 100, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, (cx, cy), 8, (200, 200, 200), 2, cv2.LINE_AA)

        # ── Label ──
        cv2.putText(
            frame, "DRAW WITH YOUR HAND (or mouse fallback)",
            (self.canvas_x, self.canvas_y + self.CANVAS_HEIGHT + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (200, 0, 200), 1, cv2.LINE_AA,
        )

        # ── OCR Result ──
        if self._cached_text:
            ry = self.canvas_y + self.CANVAS_HEIGHT + 35
            lines = self._cached_text.strip().split("\n")[:3]

            panel_h = len(lines) * 22 + 28
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (self.canvas_x, ry),
                (self.canvas_x + self.CANVAS_WIDTH, ry + panel_h),
                (40, 0, 40), -1,
            )
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            cv2.putText(frame, "[OCR Result]",
                        (self.canvas_x + 5, ry + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        config.COLOR_FPS, 1, cv2.LINE_AA)
            for i, line in enumerate(lines):
                cv2.putText(frame, line,
                            (self.canvas_x + 5, ry + 34 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            config.COLOR_WHITE, 1, cv2.LINE_AA)

    # ──────────────────────────────────────────
    # OCR Processing
    # ──────────────────────────────────────────

    def detect(self, frame):
        """
        Process hand tracking, drawing, and OCR.

        Args:
            frame: BGR video frame.

        Returns:
            Tuple of (recognized_text, [])
        """
        if not self._loaded:
            return "", []

        # Track hand and draw on canvas
        self._process_hand(frame)

        # Draw the canvas UI
        self._draw_ui(frame)

        # Frame-skipped OCR
        self._frame_count += 1
        if self._frame_count % self._process_every_n != 0:
            return self._cached_text, []

        try:
            # Check if canvas has content
            gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            if cv2.countNonZero(255 - gray) < 30:
                self._cached_text = ""
                return "", []

            # Preprocess
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            # OCR
            text = self.pytesseract.image_to_string(
                thresh, config='--oem 3 --psm 6',
            ).strip()

            self._cached_text = text if text else ""
            return self._cached_text, []

        except Exception as e:
            print(f"[HandwritingOCR] OCR error: {e}")
            return self._cached_text, []

    def is_loaded(self):
        return self._loaded
