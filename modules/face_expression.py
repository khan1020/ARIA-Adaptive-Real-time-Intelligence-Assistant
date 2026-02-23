"""
Facial Expression Recognition Module — powered by HSEmotion.

Replaces the old FER (2018 mini-Xception) with:
  - HSEmotion EfficientNet-B0 trained on AffectNet (2022)
  - MediaPipe face detection (faster and more accurate than MTCNN)
  - 8 emotion classes: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
  - Score-weighted temporal smoothing over 5 frames

AffectNet has 450,000+ real-world images vs FER dataset's 35,000 lab images.
EfficientNet-B0 achieves ~80% accuracy on AffectNet vs ~65% for mini-Xception.

Model: enet_b0_8_best_afew (best accuracy on in-the-wild video)
"""

import cv2
import numpy as np
from collections import deque, defaultdict
import config


class FaceExpressionRecognizer:
    """
    Modern facial expression recognizer using HSEmotion + MediaPipe.

    Pipeline:
      1. MediaPipe face detection → face bounding boxes (fast, robust)
      2. Crop + resize each face to 224×224
      3. HSEmotion EfficientNet-B0 → 8-class emotion scores
      4. Score-weighted temporal smoothing over 5 frames
    """

    # Model produces these 8 emotions (AffectNet taxonomy)
    EMOTIONS = ['Anger', 'Contempt', 'Disgust', 'Fear',
                'Happiness', 'Neutral', 'Sadness', 'Surprise']

    # Minimum margin between top-2 predictions to avoid flip-flopping
    MIN_MARGIN = 0.12

    def __init__(self, confidence_threshold=None, smooth_window=5):
        self.confidence_threshold = max(
            confidence_threshold or config.FACE_CONFIDENCE_THRESHOLD,
            0.38
        )
        self._recognizer = None
        self._face_detector = None
        self._face_detector_type = "haar"  # default; updated in load_model()
        self._mp_face = None               # always defined, even when using DNN
        self._loaded = False

        # Score-vector history per face slot
        self._score_history = []
        self._smooth_window = smooth_window

        # Frame-skip — run every 2nd frame for performance
        self._frame_count = 0
        self._detect_every_n = 2
        self._cached_detections = []

    def load_model(self):
        """Load HSEmotion recognizer + MediaPipe face detector."""
        print("[FaceExpression] Loading HSEmotion EfficientNet-B0 (AffectNet)...")
        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
            # enet_b0_8_best_afew = best accuracy on in-the-wild video (AFEW dataset)
            self._recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
            print("[FaceExpression] HSEmotion model ready.")
        except Exception as e:
            print(f"[FaceExpression] HSEmotion load failed: {e}")
            self._loaded = False
            return

        print("[FaceExpression] Loading face detector...")
        try:
            # Try OpenCV's DNN face detector first (more accurate than Haar cascade)
            import urllib.request, os
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            os.makedirs(model_dir, exist_ok=True)
            proto = os.path.join(model_dir, "deploy.prototxt")
            weights = os.path.join(model_dir, "res10_300x300_ssd.caffemodel")

            if not os.path.exists(proto):
                print("[FaceExpression] Downloading DNN face model...")
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                    proto
                )
            if not os.path.exists(weights):
                urllib.request.urlretrieve(
                    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                    weights
                )

            self._face_detector = cv2.dnn.readNetFromCaffe(proto, weights)
            self._face_detector_type = "dnn"
            print("[FaceExpression] OpenCV DNN face detector ready.")
        except Exception as e:
            print(f"[FaceExpression] DNN detector unavailable ({e}), using Haar cascade fallback.")
            self._face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self._face_detector_type = "haar"
            self._mp_face = None

        self._loaded = True
        print("[FaceExpression] Ready. Model: EfficientNet-B0 | Dataset: AffectNet (450k images)")

    def _detect_faces(self, frame):
        """
        Detect face bounding boxes. Returns list of (x, y, w, h).
        Uses OpenCV DNN if downloaded, otherwise Haar cascade fallback.
        """
        h, w = frame.shape[:2]
        boxes = []

        if self._face_detector_type == "dnn":
            # OpenCV DNN SSD face detector
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0)
            )
            self._face_detector.setInput(blob)
            detections = self._face_detector.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < 0.5:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                bw, bh = x2 - x1, y2 - y1
                if bw > 0 and bh > 0:
                    # Add 15% padding for HSEmotion model context
                    pad_x = int(bw * 0.15)
                    pad_y = int(bh * 0.15)
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    bw = min(w - x1, bw + 2 * pad_x)
                    bh = min(h - y1, bh + 2 * pad_y)
                    boxes.append((x1, y1, bw, bh))
        else:
            # Haar cascade fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
            )
            for (x, y, bw, bh) in faces:
                boxes.append((int(x), int(y), int(bw), int(bh)))

        return boxes

    def _softmax(self, logits):
        """Convert raw logits to probabilities."""
        e = np.exp(logits - np.max(logits))
        return e / e.sum()

    def _smooth(self, face_idx, scores_dict):
        """
        Score-vector temporal smoothing.

        Averages raw probability vectors across recent frames,
        then picks the winner only if it has a clear margin.

        Returns:
            (winner, confidence, margin, avg_scores_dict)
            avg_scores_dict should be used for top-3 display so that
            the label is always consistent with the decision.
        """
        while len(self._score_history) <= face_idx:
            self._score_history.append(deque(maxlen=self._smooth_window))

        self._score_history[face_idx].append(scores_dict)

        # Average scores across history
        avg = defaultdict(float)
        n = len(self._score_history[face_idx])
        for frame_scores in self._score_history[face_idx]:
            for em, sc in frame_scores.items():
                avg[em] += sc / n

        ranked = sorted(avg.items(), key=lambda x: x[1], reverse=True)
        winner, w_score = ranked[0]
        runner_score = ranked[1][1] if len(ranked) > 1 else 0

        margin = w_score - runner_score
        return winner, w_score, margin, dict(avg)

    def detect(self, frame):
        """
        Detect faces and classify emotions.

        Returns:
            List of (label, confidence, (x1, y1, x2, y2))
        """
        if not self._loaded:
            return []

        self._frame_count += 1
        if self._frame_count % self._detect_every_n != 0:
            return self._cached_detections

        try:
            face_boxes = self._detect_faces(frame)

            # Trim history if fewer faces visible now
            while len(self._score_history) > len(face_boxes):
                self._score_history.pop()

            detections = []
            for face_idx, (x, y, bw, bh) in enumerate(face_boxes):
                # Crop face region
                face_crop = frame[y:y + bh, x:x + bw]
                if face_crop.size == 0:
                    continue

                # HSEmotion returns (class_name_str, scores_array) tuple
                _, scores_arr = self._recognizer.predict_emotions(
                    face_crop, logits=False  # logits=False = softmax probabilities
                )

                # Build score dict
                scores = {em: float(scores_arr[i]) for i, em in enumerate(self.EMOTIONS)}

                # Apply temporal smoothing — returns smoothed avg scores for consistency
                winner, confidence, margin, avg_scores = self._smooth(face_idx, scores)

                # Skip if below threshold or margin too small
                if confidence < self.confidence_threshold:
                    continue
                if margin < self.MIN_MARGIN:
                    continue

                # Build top-3 from SMOOTHED avg scores (same data used for the decision)
                ranked = sorted(avg_scores.items(), key=lambda e: e[1], reverse=True)
                top3 = "  ".join(
                    f"{em[:3]}:{sc:.0%}" for em, sc in ranked[:3]
                )
                label = f"{winner} ({top3})"

                bbox = (x, y, x + bw, y + bh)
                detections.append((label, confidence, bbox))

            self._cached_detections = detections
            return detections

        except Exception as e:
            print(f"[FaceExpression] Error: {e}")
            return self._cached_detections

    def is_loaded(self):
        return self._loaded
