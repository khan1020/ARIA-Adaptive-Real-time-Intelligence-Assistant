"""
Face Mesh Overlay Module using MediaPipe FaceLandmarker.

Draws a detailed 478-point face mesh wireframe on detected faces.
Works alongside the emotion detection module.
"""

import cv2
import os
import config


class FaceMeshOverlay:
    """
    MediaPipe FaceLandmarker-based face mesh renderer.
    
    Draws tesselation connections across 478 facial landmarks
    for a detailed wireframe face overlay.
    """

    # Key face mesh connection groups for drawing
    # Simplified tesselation — major contours only
    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
    ]

    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
    LEFT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
    RIGHT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
    LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]
    LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
    NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]

    def __init__(self):
        self.landmarker = None
        self.mp = None
        self._loaded = False

    def load_model(self):
        """Load MediaPipe FaceLandmarker."""
        print("[FaceMesh] Loading MediaPipe FaceLandmarker...")
        try:
            import mediapipe as mp
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
                RunningMode,
            )
            from mediapipe.tasks.python import BaseOptions

            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models", "face_landmarker.task"
            )

            if not os.path.exists(model_path):
                print("[FaceMesh] Downloading face landmarker model...")
                import urllib.request
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
                    model_path,
                )
                print("[FaceMesh] Model downloaded.")

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_faces=3,
                min_face_detection_confidence=0.4,
                min_face_presence_confidence=0.4,
                min_tracking_confidence=0.3,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )

            self.landmarker = FaceLandmarker.create_from_options(options)
            self.mp = mp
            self._loaded = True
            print("[FaceMesh] FaceLandmarker loaded successfully.")
        except Exception as e:
            print(f"[FaceMesh] Error loading: {e}")
            import traceback
            traceback.print_exc()
            self._loaded = False

    def _draw_contour(self, frame, landmarks, indices, color, h, w, thickness=1):
        """Draw a contour connecting landmark indices."""
        for i in range(len(indices) - 1):
            idx1 = indices[i]
            idx2 = indices[i + 1]
            if idx1 < len(landmarks) and idx2 < len(landmarks):
                pt1 = (int(landmarks[idx1].x * w), int(landmarks[idx1].y * h))
                pt2 = (int(landmarks[idx2].x * w), int(landmarks[idx2].y * h))
                cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

    def _draw_mesh_points(self, frame, landmarks, h, w):
        """Draw sparse mesh points across the face."""
        # Draw every 3rd landmark as a small dot for the mesh effect
        for i in range(0, len(landmarks), 3):
            lm = landmarks[i]
            px = int(lm.x * w)
            py = int(lm.y * h)
            cv2.circle(frame, (px, py), 1, (100, 255, 200), -1, cv2.LINE_AA)

    def detect(self, frame):
        """
        Detect faces and draw mesh overlay.

        Args:
            frame: BGR image.

        Returns:
            Tuple of (face_count, list_of_face_landmarks).
            Each face_landmarks is a list of 478 landmarks.
        """
        if not self._loaded:
            return 0, []

        try:
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)

            result = self.landmarker.detect(mp_image)

            if not result.face_landmarks:
                return 0, []

            all_landmarks = []
            for face_landmarks in result.face_landmarks:
                lms = face_landmarks
                all_landmarks.append(lms)

                # Mesh dots
                self._draw_mesh_points(frame, lms, h, w)

                # Face oval
                self._draw_contour(frame, lms, self.FACE_OVAL, (0, 255, 200), h, w, 1)

                # Eyes
                self._draw_contour(frame, lms, self.LEFT_EYE, (0, 200, 255), h, w, 1)
                self._draw_contour(frame, lms, self.RIGHT_EYE, (0, 200, 255), h, w, 1)

                # Eyebrows
                self._draw_contour(frame, lms, self.LEFT_EYEBROW, (255, 200, 0), h, w, 1)
                self._draw_contour(frame, lms, self.RIGHT_EYEBROW, (255, 200, 0), h, w, 1)

                # Lips
                self._draw_contour(frame, lms, self.LIPS_OUTER, (0, 100, 255), h, w, 1)
                self._draw_contour(frame, lms, self.LIPS_INNER, (0, 50, 200), h, w, 1)

                # Nose
                self._draw_contour(frame, lms, self.NOSE, (200, 200, 200), h, w, 1)

            return len(result.face_landmarks), all_landmarks

        except Exception as e:
            print(f"[FaceMesh] Error: {e}")
            return 0, []

    def is_loaded(self):
        return self._loaded

    def release(self):
        if self.landmarker:
            self.landmarker.close()
