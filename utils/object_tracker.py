"""
Smart Object Tracker — IoU-based tracking with unique IDs and motion trails.

Assigns persistent IDs to detected objects across frames using
Intersection over Union (IoU) matching. Maintains motion trails.
"""

import cv2
import time


class TrackedObject:
    """A single tracked object with ID, history, and trail."""

    _next_id = 1

    def __init__(self, label, confidence, bbox):
        self.id = TrackedObject._next_id
        TrackedObject._next_id += 1
        self.label = label
        self.confidence = confidence
        self.bbox = bbox
        self.trail = [self._center(bbox)]
        self.max_trail = 30
        self.last_seen = time.time()
        self.frames_alive = 1

    def _center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, label, confidence, bbox):
        self.label = label
        self.confidence = confidence
        self.bbox = bbox
        self.trail.append(self._center(bbox))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)
        self.last_seen = time.time()
        self.frames_alive += 1

    def center(self):
        return self._center(self.bbox)


class ObjectTracker:
    """
    IoU-based multi-object tracker.
    
    Matches new detections to existing tracks using IoU.
    Assigns unique IDs and maintains motion trail history.
    """

    def __init__(self, iou_threshold=0.3, max_lost_seconds=1.0):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost_seconds
        self.tracked = []  # List of TrackedObject

    def _iou(self, box_a, box_b):
        """Calculate Intersection over Union."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0

    def update(self, detections):
        """
        Match new detections to existing tracks.

        Args:
            detections: List of (label, confidence, (x1,y1,x2,y2))

        Returns:
            List of TrackedObject with updated positions.
        """
        now = time.time()

        # Remove stale tracks
        self.tracked = [t for t in self.tracked if (now - t.last_seen) < self.max_lost]

        if not detections:
            return self.tracked

        matched_tracks = set()
        matched_dets = set()

        # Match detections to existing tracks by IoU
        for di, (label, conf, bbox) in enumerate(detections):
            best_iou = 0
            best_ti = -1

            for ti, track in enumerate(self.tracked):
                if ti in matched_tracks:
                    continue
                if track.label != label:
                    continue
                iou = self._iou(track.bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_ti = ti

            if best_iou >= self.iou_threshold and best_ti >= 0:
                self.tracked[best_ti].update(label, conf, bbox)
                matched_tracks.add(best_ti)
                matched_dets.add(di)

        # Create new tracks for unmatched detections
        for di, (label, conf, bbox) in enumerate(detections):
            if di not in matched_dets:
                self.tracked.append(TrackedObject(label, conf, bbox))

        return self.tracked

    def draw_trails(self, frame):
        """Draw motion trails for all tracked objects."""
        trail_colors = [
            (0, 255, 200), (255, 100, 0), (0, 200, 255),
            (200, 0, 255), (255, 255, 0), (100, 255, 100),
        ]

        for track in self.tracked:
            if len(track.trail) < 2:
                continue

            color = trail_colors[track.id % len(trail_colors)]

            # Draw trail with fading
            for i in range(1, len(track.trail)):
                alpha = i / len(track.trail)
                thickness = max(1, int(3 * alpha))
                pt1 = track.trail[i - 1]
                pt2 = track.trail[i]
                c = tuple(int(v * alpha) for v in color)
                cv2.line(frame, pt1, pt2, c, thickness, cv2.LINE_AA)

            # Draw ID label at current position
            cx, cy = track.center()
            label = f"ID:{track.id} {track.label}"
            cv2.putText(
                frame, label, (cx - 20, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                color, 1, cv2.LINE_AA,
            )

    def get_object_counts(self):
        """Get counts of each object type currently tracked."""
        counts = {}
        for t in self.tracked:
            counts[t.label] = counts.get(t.label, 0) + 1
        return counts
