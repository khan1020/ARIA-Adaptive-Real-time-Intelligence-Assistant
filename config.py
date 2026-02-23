"""
Configuration settings for the Multimodal AI Vision Recognition System.
"""

import os

# ──────────────────────────────────────────────
# Camera Settings
# ──────────────────────────────────────────────
# Set to 0 for laptop webcam, or a URL for phone camera:
#   Android IP Webcam:  "http://192.168.1.X:8080/video"
#   DroidCam:           "http://192.168.1.X:4747/video"
#   EpocCam (iOS):      works as USB webcam (use 0 or 1)
CAMERA_SOURCE = 0
# CAMERA_SOURCE = "http://192.168.100.121:8080/video"  # IP Webcam app URL
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ──────────────────────────────────────────────
# Tesseract OCR Path (Windows default)
# ──────────────────────────────────────────────
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ──────────────────────────────────────────────
# Confidence Thresholds
# ──────────────────────────────────────────────
YOLO_CONFIDENCE_THRESHOLD = 0.4   # Min confidence to show an object (0.0-1.0)
YOLO_IOU_THRESHOLD = 0.45         # NMS overlap threshold — lower = fewer overlapping boxes
FACE_CONFIDENCE_THRESHOLD = 0.3
HAND_CONFIDENCE_THRESHOLD = 0.5
OCR_CONFIDENCE_THRESHOLD = 30  # Tesseract confidence (0-100)

# ──────────────────────────────────────────────
# Colors (BGR format for OpenCV)
# ──────────────────────────────────────────────
COLOR_OBJECT_DETECTION = (0, 255, 0)       # Green
COLOR_FACE_EXPRESSION = (255, 100, 50)     # Blue-ish
COLOR_HAND_GESTURE = (0, 200, 255)         # Orange-Yellow
COLOR_OCR_TEXT = (255, 0, 255)             # Magenta
COLOR_FPS = (0, 255, 255)                  # Yellow
COLOR_PANEL_BG = (30, 30, 30)             # Dark gray
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# ──────────────────────────────────────────────
# Module Display Names
# ──────────────────────────────────────────────
MODULE_NAMES = {
    1: "Object Detection + Segmentation",
    2: "Facial Expression Recognition",
    3: "Hand Gesture Recognition",
    4: "Handwritten Text Recognition (OCR)",
    5: "Face Mesh Overlay",
    6: "Gesture-Controlled PC",
    7: "Drowsiness/Attention Detection",
}

COLOR_FACE_MESH = (0, 255, 200)            # Cyan-green

# ──────────────────────────────────────────────
# Font Settings
# ──────────────────────────────────────────────
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# ──────────────────────────────────────────────
# Performance
# ──────────────────────────────────────────────
FPS_ROLLING_WINDOW = 30  # Number of frames for rolling FPS average
