"""
Threaded Camera Reader — true real-time for IP Webcam streams.

For IP cameras: uses the /shot.jpg snapshot endpoint instead of
the MJPEG /video stream. Each request returns the CURRENT frame
with zero buffering — no lag at all.

For local webcams: uses cv2.VideoCapture with grab() draining.
"""

import cv2
import threading
import time
import numpy as np


class CameraReader:
    """
    Real-time camera reader.

    IP cameras (URLs): Fetches individual JPEG snapshots from
    the /shot.jpg endpoint in a background thread. Each fetch
    returns the absolute latest frame — no buffer, no lag.

    Local cameras: Uses OpenCV VideoCapture with a background
    thread that continuously grab()s to drain the buffer.
    """

    def __init__(self, source, width=None, height=None):
        self.source = source
        self.width = width
        self.height = height

        self.frame = None
        self.ret = False
        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self._is_url = isinstance(source, str)
        self._cap = None
        self._snapshot_url = None

    def open(self):
        """Open camera and start reader thread."""
        if self._is_url:
            # Build snapshot URL from video URL
            # "http://192.168.100.121:8080/video" -> "http://192.168.100.121:8080/shot.jpg"
            base = self.source.rsplit('/', 1)[0]
            self._snapshot_url = base + "/shot.jpg"
            print(f"[Camera] Using snapshot endpoint: {self._snapshot_url}")

            # Test connection with first snapshot
            try:
                import urllib.request
                resp = urllib.request.urlopen(self._snapshot_url, timeout=5)
                data = resp.read()
                arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    print("[Camera] Snapshot decode failed, falling back to video stream")
                    return self._open_video_stream()
                with self._lock:
                    self.frame = frame
                    self.ret = True
                print(f"[Camera] Snapshot OK: {frame.shape[1]}x{frame.shape[0]}")
            except Exception as e:
                print(f"[Camera] Snapshot failed ({e}), falling back to video stream")
                return self._open_video_stream()

        else:
            # Local webcam
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                return False
            if self.width:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            ret, frame = self._cap.read()
            if not ret:
                return False
            with self._lock:
                self.frame = frame
                self.ret = True

        # Start background reader
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return True

    def _open_video_stream(self):
        """Fallback: open MJPEG video stream with buffer draining."""
        self._snapshot_url = None  # Disable snapshot mode
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = self._cap.read()
        if not ret:
            return False
        with self._lock:
            self.frame = frame
            self.ret = True
        return True

    def _reader(self):
        """Background thread: fetch latest frame continuously."""
        if self._snapshot_url:
            self._reader_snapshot()
        elif self._cap:
            self._reader_videocap()

    def _reader_snapshot(self):
        """Snapshot reader: fetch /shot.jpg in a loop."""
        import urllib.request

        while self._running:
            try:
                resp = urllib.request.urlopen(self._snapshot_url, timeout=3)
                data = resp.read()
                arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with self._lock:
                        self.frame = frame
                        self.ret = True
            except Exception:
                time.sleep(0.05)

    def _reader_videocap(self):
        """VideoCapture reader: grab() to drain buffer."""
        while self._running:
            ret = self._cap.grab()
            if not ret:
                time.sleep(0.01)
                continue
            ret, frame = self._cap.retrieve()
            if ret and frame is not None:
                with self._lock:
                    self.frame = frame
                    self.ret = True

    def read(self):
        """Get the latest frame (non-blocking)."""
        with self._lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def get_resolution(self):
        """Get actual frame resolution."""
        with self._lock:
            if self.frame is not None:
                h, w = self.frame.shape[:2]
                return w, h
        return 0, 0

    def is_url_source(self):
        return self._is_url

    def release(self):
        """Stop reader and release resources."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
