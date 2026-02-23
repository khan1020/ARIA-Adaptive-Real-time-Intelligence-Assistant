"""
Drawing utilities for rendering bounding boxes, labels, and info panels
on video frames.
"""

import cv2
import numpy as np
import config


def draw_bbox(frame, bbox, label, confidence, color):
    """
    Draw a bounding box with a label and confidence score.

    Args:
        frame: The video frame (numpy array).
        bbox: Tuple of (x1, y1, x2, y2).
        label: Text label for the detection.
        confidence: Confidence score (0.0 - 1.0).
        color: BGR color tuple.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Prepare label text
    text = f"{label}: {confidence:.0%}" if confidence else label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = config.FONT_SCALE
    thickness = config.FONT_THICKNESS
    
    # Get text size for background rectangle
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw label background
    cv2.rectangle(
        frame,
        (x1, y1 - text_h - 10),
        (x1 + text_w + 8, y1),
        color,
        -1,
    )
    
    # Draw label text
    cv2.putText(
        frame,
        text,
        (x1 + 4, y1 - 6),
        font,
        font_scale,
        config.COLOR_WHITE,
        thickness,
        cv2.LINE_AA,
    )


def draw_text_panel(frame, lines, position="top-left", bg_alpha=0.7):
    """
    Draw a semi-transparent panel with text lines.

    Args:
        frame: The video frame.
        lines: List of strings to display.
        position: 'top-left', 'top-right', 'bottom-left', or 'bottom-right'.
        bg_alpha: Opacity of the background (0.0 - 1.0).
    """
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    line_height = 28
    padding = 12

    # Calculate panel dimensions
    max_width = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, w)

    panel_w = max_width + padding * 2
    panel_h = len(lines) * line_height + padding * 2

    h, w = frame.shape[:2]

    # Calculate position
    if position == "top-left":
        x, y = 10, 10
    elif position == "top-right":
        x, y = w - panel_w - 10, 10
    elif position == "bottom-left":
        x, y = 10, h - panel_h - 10
    elif position == "bottom-right":
        x, y = w - panel_w - 10, h - panel_h - 10
    else:
        x, y = 10, 10

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), config.COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)

    # Draw border
    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), (80, 80, 80), 1)

    # Draw text lines
    for i, line in enumerate(lines):
        text_y = y + padding + (i + 1) * line_height - 8
        cv2.putText(
            frame, line, (x + padding, text_y),
            font, font_scale, config.COLOR_WHITE, thickness, cv2.LINE_AA
        )


def draw_mode_indicator(frame, active_modes):
    """
    Draw indicators showing which modules are currently active.

    Args:
        frame: The video frame.
        active_modes: Dict mapping module number to active state (bool).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    h, w = frame.shape[:2]
    
    y_start = h - 20
    x = 10

    for num, name in config.MODULE_NAMES.items():
        is_active = active_modes.get(num, False)
        color = (0, 255, 0) if is_active else (80, 80, 80)
        status = "ON" if is_active else "OFF"
        text = f"[{num}] {name}: {status}"
        
        cv2.putText(
            frame, text, (x, y_start),
            font, font_scale, color, thickness, cv2.LINE_AA
        )
        y_start -= 24


def draw_ocr_result(frame, text, region_bbox=None):
    """
    Draw recognized text from OCR on the frame.

    Args:
        frame: The video frame.
        text: Recognized text string.
        region_bbox: Optional bounding box around the text region.
    """
    if not text or not text.strip():
        return

    if region_bbox:
        x1, y1, x2, y2 = [int(v) for v in region_bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_OCR_TEXT, 2)

    # Draw OCR text in a panel at the bottom
    lines = ["[OCR Result]"] + text.strip().split("\n")[:5]  # Max 5 lines
    draw_text_panel(frame, lines, position="bottom-right")
