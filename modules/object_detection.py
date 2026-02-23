"""
Object Detection & Segmentation Module using Ultralytics YOLOv8.

Uses YOLOv8n-seg for real-time object detection with pixel-perfect
instance segmentation masks. Falls back to regular detection if
segmentation model fails to load.
"""

import cv2
import numpy as np
import config


class ObjectDetector:
    """
    YOLO-based object detector with segmentation support.

    Uses YOLOv8n-seg which provides both bounding boxes and
    pixel-perfect segmentation masks for each detected object.
    """

    # Colors for segmentation masks (BGR)
    MASK_COLORS = [
        (255, 100, 50),   # blue
        (50, 255, 100),   # green
        (100, 50, 255),   # red
        (255, 200, 0),    # cyan
        (0, 200, 255),    # yellow
        (200, 0, 255),    # magenta
        (255, 150, 150),  # light blue
        (150, 255, 150),  # light green
        (150, 150, 255),  # light red
        (0, 255, 255),    # yellow
    ]

    def __init__(self, confidence_threshold=None):
        self.confidence_threshold = confidence_threshold or config.YOLO_CONFIDENCE_THRESHOLD
        self.model = None
        self._loaded = False
        self._has_seg = False

    def load_model(self):
        """Load the YOLOv8n-seg model (segmentation)."""
        print("[ObjectDetector] Loading YOLOv8n-seg model...")
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n-seg.pt")  # Auto-downloads, ~7MB
            self._loaded = True
            self._has_seg = True
            print("[ObjectDetector] Segmentation model loaded successfully.")
        except Exception as e:
            print(f"[ObjectDetector] Seg model failed, trying regular: {e}")
            try:
                from ultralytics import YOLO
                self.model = YOLO("yolov8n.pt")
                self._loaded = True
                self._has_seg = False
                print("[ObjectDetector] Regular model loaded (no segmentation).")
            except Exception as e2:
                print(f"[ObjectDetector] Error loading model: {e2}")
                self._loaded = False

    def detect(self, frame):
        """
        Run object detection + segmentation on a frame.

        Args:
            frame: BGR image.

        Returns:
            List of tuples: (label, confidence, (x1, y1, x2, y2))
        """
        if not self._loaded:
            return []

        try:
            results = self.model(
                frame, verbose=False,
                conf=self.confidence_threshold,
                iou=config.YOLO_IOU_THRESHOLD,
                agnostic_nms=True,
                imgsz=640,
            )

            detections = []
            for result in results:
                boxes = result.boxes

                # ── Draw segmentation masks ──
                if self._has_seg and result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    h, w = frame.shape[:2]

                    for idx, mask in enumerate(masks):
                        # Resize mask to frame size
                        mask_resized = cv2.resize(mask, (w, h))
                        mask_bool = mask_resized > 0.5

                        # Get color for this instance
                        color = self.MASK_COLORS[idx % len(self.MASK_COLORS)]

                        # Create colored overlay
                        colored_mask = np.zeros_like(frame)
                        colored_mask[mask_bool] = color

                        # Blend with frame (semi-transparent)
                        frame[mask_bool] = cv2.addWeighted(
                            frame, 0.6, colored_mask, 0.4, 0
                        )[mask_bool]

                        # Draw mask contour for clean edges
                        contours, _ = cv2.findContours(
                            (mask_resized > 0.5).astype(np.uint8),
                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(frame, contours, -1, color, 2, cv2.LINE_AA)

                # ── Parse bounding boxes ──
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = result.names[class_id]
                    detections.append((label, confidence, (x1, y1, x2, y2)))

            return detections

        except Exception as e:
            print(f"[ObjectDetector] Detection error: {e}")
            return []

    def is_loaded(self):
        return self._loaded
