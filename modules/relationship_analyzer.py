"""
Multi-Object Relationship Analyzer.

Uses spatial proximity between bounding boxes to infer
relationships like "person holding cup", "person near laptop".
"""


class RelationshipAnalyzer:
    """
    Spatial proximity-based relationship inference.
    
    Analyzes bounding box overlaps and distances to generate
    natural language relationship descriptions.
    """

    # Relationship rules: (obj_a, obj_b, relation_verb)
    RELATIONSHIP_RULES = [
        ("person", "cell phone", "using"),
        ("person", "laptop", "using"),
        ("person", "cup", "holding"),
        ("person", "bottle", "holding"),
        ("person", "book", "reading"),
        ("person", "remote", "holding"),
        ("person", "mouse", "using"),
        ("person", "keyboard", "using"),
        ("person", "tv", "watching"),
        ("person", "monitor", "looking at"),
        ("person", "chair", "sitting on"),
        ("person", "backpack", "wearing"),
        ("person", "handbag", "carrying"),
        ("person", "umbrella", "holding"),
        ("person", "sports ball", "playing with"),
        ("person", "dog", "with"),
        ("person", "cat", "with"),
    ]

    def __init__(self, proximity_threshold=0.4):
        """
        Args:
            proximity_threshold: Max normalized distance for "near" relationship.
        """
        self.proximity_threshold = proximity_threshold
        self._cached_relations = []

    def _center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _overlap(self, bbox_a, bbox_b):
        """Check if two bounding boxes overlap."""
        x1 = max(bbox_a[0], bbox_b[0])
        y1 = max(bbox_a[1], bbox_b[1])
        x2 = min(bbox_a[2], bbox_b[2])
        y2 = min(bbox_a[3], bbox_b[3])
        return x2 > x1 and y2 > y1

    def _proximity(self, bbox_a, bbox_b, frame_w):
        """Normalized distance between bbox centers."""
        ca = self._center(bbox_a)
        cb = self._center(bbox_b)
        dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
        return dist / max(frame_w, 1)

    def analyze(self, detections, frame_width=640):
        """
        Analyze spatial relationships between detections.

        Args:
            detections: List of (label, confidence, (x1,y1,x2,y2))
            frame_width: Width of the frame for normalization.

        Returns:
            List of relationship strings.
        """
        if not detections or len(detections) < 2:
            self._cached_relations = []
            return []

        relations = []

        for i, (label_a, conf_a, bbox_a) in enumerate(detections):
            for j, (label_b, conf_b, bbox_b) in enumerate(detections):
                if i >= j:
                    continue

                la = label_a.lower()
                lb = label_b.lower()

                # Check predefined relationship rules
                for obj_a, obj_b, verb in self.RELATIONSHIP_RULES:
                    matched = False
                    if la == obj_a and lb == obj_b:
                        matched = True
                    elif la == obj_b and lb == obj_a:
                        matched = True
                        # Swap to maintain "person verb object" order
                        la, lb = lb, la
                        bbox_a, bbox_b = bbox_b, bbox_a

                    if matched:
                        # Check proximity
                        if self._overlap(bbox_a, bbox_b):
                            relations.append(f"Person {verb} {obj_b}")
                        elif self._proximity(bbox_a, bbox_b, frame_width) < self.proximity_threshold:
                            relations.append(f"Person near {obj_b}")
                        break

                # Generic proximity for non-predefined pairs
                if la != lb and la != "person" and lb != "person":
                    if self._overlap(bbox_a, bbox_b):
                        relations.append(f"{label_a} near {label_b}")

        self._cached_relations = relations[:5]
        return self._cached_relations

    def get_relations(self):
        return self._cached_relations
