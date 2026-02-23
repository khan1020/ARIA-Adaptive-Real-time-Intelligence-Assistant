"""
Multimodal AI Based Real Time Vision Recognition System
=======================================================

Main entry point that integrates all AI modules into a unified
real-time pipeline using OpenCV webcam input.

Controls:
    1 - Toggle Object Detection + Segmentation
    2 - Toggle Facial Expression Recognition
    3 - Toggle Hand Gesture Recognition
    4 - Toggle Handwritten Text Recognition (OCR)
    5 - Toggle Face Mesh Overlay
    6 - Toggle Gesture-Controlled Computer
    7 - Toggle Drowsiness/Attention Detection
    V - Toggle Voice Feedback
    A - Enable All modules
    N - Disable All modules
    Q - Quit

Author: Fareed
"""

import cv2
import sys
import config

# ── Core AI Modules ──
from modules.object_detection import ObjectDetector
from modules.face_expression import FaceExpressionRecognizer
from modules.hand_gesture import HandGestureRecognizer
from modules.handwriting_ocr import HandwritingOCR
from modules.face_mesh import FaceMeshOverlay
from modules.gesture_control import GestureController
from modules.drowsiness_detector import DrowsinessDetector

# ── AI Enhancement Modules ──
from modules.voice_feedback import VoiceFeedback
from modules.scene_describer import SceneDescriber
from modules.activity_recognition import ActivityRecognizer
from modules.relationship_analyzer import RelationshipAnalyzer

# ── Utilities ──
from utils.drawing import draw_bbox, draw_text_panel, draw_mode_indicator
from utils.performance import FPSCounter, LatencyTracker
from utils.object_tracker import ObjectTracker
from utils.ai_dashboard import AIDashboard
from utils.camera_reader import CameraReader


# ──────────────────────────────────────────────
# Window name
# ──────────────────────────────────────────────
WINDOW_NAME = "Multimodal AI Vision System"


def print_banner():
    """Print the startup banner with controls."""
    print("=" * 60)
    print("  Multimodal AI Based Real Time Vision Recognition System")
    print("=" * 60)
    print()
    print("  Controls:")
    print("    [1] Toggle Object Detection + Segmentation")
    print("    [2] Toggle Facial Expression Recognition")
    print("    [3] Toggle Hand Gesture Recognition")
    print("    [4] Toggle Handwritten Text Recognition (OCR)")
    print("    [5] Toggle Face Mesh Overlay")
    print("    [6] Toggle Gesture-Controlled Computer")
    print("    [7] Toggle Drowsiness/Attention Detection")
    print("    [V] Toggle Voice Feedback")
    print("    [R] Rotate Camera (0° → 90° → 180° → 270°)")
    print("    [A] Enable ALL modules")
    print("    [N] Disable ALL modules")
    print("    [Q] Quit")
    print()
    print("-" * 60)


def initialize_modules():
    """Initialize and load all AI modules."""
    modules = {
        1: ObjectDetector(),
        2: FaceExpressionRecognizer(),
        3: HandGestureRecognizer(),
        4: HandwritingOCR(),
        5: FaceMeshOverlay(),
        6: GestureController(),
        7: DrowsinessDetector(),
    }

    print("\nInitializing AI modules...\n")
    for num, module in modules.items():
        name = config.MODULE_NAMES.get(num, f"Module {num}")
        print(f"  Loading {name}...")
        if num == 6:
            module.load()
        elif num == 7:
            pass  # Drowsiness uses face mesh, no separate model
        else:
            module.load_model()
        status = "✓ Ready"
        if num == 7:
            status = "✓ Ready (uses Face Mesh)"
        elif hasattr(module, 'is_loaded') and not module.is_loaded():
            status = "✗ Failed"
        print(f"  {status}\n")

    return modules


def initialize_ai_features():
    """Initialize all AI enhancement modules."""
    print("Initializing AI enhancement features...\n")

    voice = VoiceFeedback(cooldown_seconds=4)
    voice.load()

    tracker = ObjectTracker(iou_threshold=0.3, max_lost_seconds=1.0)
    dashboard = AIDashboard(max_log_entries=8)
    describer = SceneDescriber(update_interval=2.0)
    activity = ActivityRecognizer(persistence_seconds=1.0)
    relationships = RelationshipAnalyzer(proximity_threshold=0.4)

    print("  AI features initialized.\n")

    return {
        "voice": voice,
        "tracker": tracker,
        "dashboard": dashboard,
        "describer": describer,
        "activity": activity,
        "relationships": relationships,
    }


def process_frame(frame, modules, active_modes, latency_tracker, ai_features):
    """
    Process a single frame through all active modules.
    """
    colors = {
        1: config.COLOR_OBJECT_DETECTION,
        2: config.COLOR_FACE_EXPRESSION,
        3: config.COLOR_HAND_GESTURE,
        4: config.COLOR_OCR_TEXT,
        5: (0, 255, 200),
    }

    # Collect detections from all modules for AI features
    obj_detections = []
    emo_detections = []
    gest_detections = []
    ocr_text = ""
    face_landmarks_list = []
    hand_raw_landmarks = None

    # ── Object Detection + Segmentation ──
    if active_modes.get(1) and modules[1].is_loaded():
        latency_tracker.start("Objects")
        obj_detections = modules[1].detect(frame)
        latency_tracker.stop("Objects")
        for label, conf, bbox in obj_detections:
            draw_bbox(frame, bbox, label, conf, colors[1])

    # ── Facial Expression ──
    if active_modes.get(2) and modules[2].is_loaded():
        latency_tracker.start("Faces")
        emo_detections = modules[2].detect(frame)
        latency_tracker.stop("Faces")
        for emotion, conf, bbox in emo_detections:
            draw_bbox(frame, bbox, emotion.capitalize(), conf, colors[2])

    # ── Hand Gesture ──
    if active_modes.get(3) and modules[3].is_loaded():
        latency_tracker.start("Hands")
        gest_detections = modules[3].detect(frame)
        latency_tracker.stop("Hands")
        for gesture, conf, bbox in gest_detections:
            draw_bbox(frame, bbox, gesture, conf, colors[3])

        # Get raw landmarks from the hand module's last detection
        if hasattr(modules[3], '_last_landmarks') and modules[3]._last_landmarks:
            hand_raw_landmarks = modules[3]._last_landmarks

    # ── Handwritten Text OCR ──
    if active_modes.get(4) and modules[4].is_loaded():
        latency_tracker.start("OCR")
        ocr_text, _ = modules[4].detect(frame)
        latency_tracker.stop("OCR")

    # ── Face Mesh Overlay ──
    if active_modes.get(5) and modules[5].is_loaded():
        latency_tracker.start("Mesh")
        face_count, face_landmarks_list = modules[5].detect(frame)
        latency_tracker.stop("Mesh")

    # ── Gesture-Controlled Computer ──
    if active_modes.get(6) and modules[6].is_loaded():
        action = modules[6].process(gest_detections, hand_raw_landmarks)
        modules[6].draw(frame)

    # ── Drowsiness/Attention Detection ──
    if active_modes.get(7):
        # Reuse already-computed face_landmarks_list if module 5 ran this frame.
        # Only run a separate mesh detection if module 5 is OFF — avoid double work.
        lms = None
        if face_landmarks_list:
            # Module 5 already ran — reuse its landmarks, no extra detection needed
            lms = face_landmarks_list[0]
        elif modules[5].is_loaded() and not active_modes.get(5):
            # Module 5 is OFF but loaded — run it once just to get landmarks
            latency_tracker.start("Drowsy-Mesh")
            _, temp_lms = modules[5].detect(frame)
            latency_tracker.stop("Drowsy-Mesh")
            if temp_lms:
                lms = temp_lms[0]

        state = modules[7].update(lms)
        modules[7].draw(frame)

        # Drowsy voice alert
        voice = ai_features["voice"]
        if modules[7].should_alert() and voice.is_enabled():
            voice.announce("drowsy", "Warning! Drowsiness detected! Please wake up!")

    # ══════════════════════════════════════════
    # AI Enhancement Features
    # ══════════════════════════════════════════

    tracker = ai_features["tracker"]
    dashboard = ai_features["dashboard"]
    voice = ai_features["voice"]
    describer = ai_features["describer"]
    activity = ai_features["activity"]
    relationships = ai_features["relationships"]

    # ── Object Tracking ──
    if obj_detections:
        tracked = tracker.update(obj_detections)
        tracker.draw_trails(frame)
        dashboard.update_counts(tracker.get_object_counts())

    # ── Relationship Analysis ──
    relations = []
    if obj_detections and len(obj_detections) >= 2:
        frame_w = frame.shape[1]
        relations = relationships.analyze(obj_detections, frame_w)

    # ── Activity Recognition ──
    activity_str = activity.update(
        object_detections=obj_detections,
        gesture_detections=gest_detections,
        emotion_detections=emo_detections,
        ocr_text=ocr_text,
    )
    activity.draw(frame)

    # ── Scene Description ──
    describer.update(
        object_detections=obj_detections,
        emotion_detections=emo_detections,
        gesture_detections=gest_detections,
        ocr_text=ocr_text,
        relationships=relations,
        activity=activity_str,
    )
    describer.draw(frame)

    # ── Dashboard Logging ──
    if obj_detections:
        dashboard.log_detections("object", obj_detections)
    if emo_detections:
        dashboard.log_detections("emotion", emo_detections)
    if gest_detections:
        dashboard.log_detections("gesture", gest_detections)
    if relations:
        for r in relations[:1]:
            dashboard.add_detection("relation", r, 0.8)

    # ── Voice Feedback ──
    if voice.is_enabled():
        voice.announce_objects(obj_detections)
        voice.announce_emotion(emo_detections)
        voice.announce_gesture(gest_detections)
        if ocr_text:
            voice.announce_ocr(ocr_text)
        voice.announce_activity(activity_str)
        voice.announce_relations(relations)

    return frame


def build_info_lines(fps, active_modes, latency_tracker, voice_enabled):
    """Build the info panel text lines."""
    lines = [
        f"FPS: {fps:.1f}",
        "",
    ]

    active_count = sum(1 for v in active_modes.values() if v)
    lines.append(f"Active Modules: {active_count}/7")
    lines.append(f"Voice: {'ON' if voice_enabled else 'OFF'}")
    lines.append("")

    latencies = latency_tracker.get_all_latencies()
    if latencies:
        lines.append("Latency:")
        for name, ms in latencies.items():
            lines.append(f"  {name}: {ms:.1f} ms")

    return lines


def main():
    """Main application loop."""
    print_banner()

    # ── Initialize modules ──
    modules = initialize_modules()
    ai_features = initialize_ai_features()

    # ── Module states (all off by default) ──
    active_modes = {1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False}

    # ── Rotation state: 0=none, 1=90° CW, 2=180°, 3=90° CCW ──
    rotation_state = 0
    ROTATION_LABELS = ["0°", "90°", "180°", "270°"]

    # ── Performance trackers ──
    fps_counter = FPSCounter()
    latency_tracker = LatencyTracker()

    # ── Mouse callback for drawing canvas ──
    def mouse_callback(event, x, y, flags, param):
        if active_modes.get(4) and modules[4].is_loaded():
            modules[4].handle_mouse(event, x, y, flags, param)

    # ── Open camera (threaded reader for zero-latency IP streams) ──
    source = config.CAMERA_SOURCE
    if isinstance(source, str):
        print(f"Opening phone camera: {source}")
    else:
        print(f"Opening laptop camera (index {source})")

    cam = CameraReader(source, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT)

    if not cam.open():
        print("ERROR: Could not open camera!")
        if isinstance(source, str):
            print(f"Check that the URL is correct: {source}")
            print("Make sure your phone and laptop are on the same WiFi.")
        else:
            print("Please check that a camera is connected and not in use.")
        sys.exit(1)

    actual_w, actual_h = cam.get_resolution()
    is_phone = cam.is_url_source()
    print(f"Camera opened: {actual_w}x{actual_h}" + (" (phone - threaded reader)" if is_phone else ""))
    print("\nSystem ready! Press keys to toggle modules.\n")

    # Register mouse callback
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    # ── Main loop ──
    consecutive_failures = 0
    MAX_FAILURES = 30  # Allow up to 30 consecutive bad frames before quitting

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    print("ERROR: Camera stopped responding after 30 consecutive failures.")
                    break
                continue  # Skip this frame, try again
            consecutive_failures = 0  # Reset on good frame

            # Manual rotation (R key toggles)
            if rotation_state == 1:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_state == 2:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation_state == 3:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Flip frame horizontally
            frame = cv2.flip(frame, 1)

            # Tick FPS counter
            fps_counter.tick()
            fps = fps_counter.get_fps()
            ai_features["dashboard"].update_fps(fps)

            # Process frame through all active modules
            frame = process_frame(frame, modules, active_modes, latency_tracker, ai_features)

            # Draw info panel (top-left)
            voice_on = ai_features["voice"].is_enabled()
            info_lines = build_info_lines(fps, active_modes, latency_tracker, voice_on)
            draw_text_panel(frame, info_lines, position="top-left")

            # Draw AI dashboard (left side, below info)
            ai_features["dashboard"].draw(frame)

            # Draw mode indicators (bottom)
            draw_mode_indicator(frame, active_modes)

            # Draw rotation indicator (top-right corner)
            fh, fw = frame.shape[:2]
            rot_text = f"R: {ROTATION_LABELS[rotation_state]}"
            overlay_r = frame.copy()
            cv2.rectangle(overlay_r, (fw - 75, 5), (fw - 5, 30), (30, 30, 40), -1)
            cv2.addWeighted(overlay_r, 0.7, frame, 0.3, 0, frame)
            rot_color = (100, 255, 200) if rotation_state > 0 else (150, 150, 150)
            cv2.putText(frame, rot_text, (fw - 70, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, rot_color, 1, cv2.LINE_AA)

            # Show frame
            cv2.imshow(WINDOW_NAME, frame)

            # ── Handle keyboard input ──
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print("\nQuitting...")
                break

            elif key == ord('1'):
                active_modes[1] = not active_modes[1]
                status = "ON" if active_modes[1] else "OFF"
                print(f"  Object Detection + Segmentation: {status}")

            elif key == ord('2'):
                active_modes[2] = not active_modes[2]
                status = "ON" if active_modes[2] else "OFF"
                print(f"  Facial Expression: {status}")

            elif key == ord('3'):
                active_modes[3] = not active_modes[3]
                status = "ON" if active_modes[3] else "OFF"
                print(f"  Hand Gesture: {status}")

            elif key == ord('4'):
                active_modes[4] = not active_modes[4]
                status = "ON" if active_modes[4] else "OFF"
                print(f"  Handwritten Text OCR: {status}")

            elif key == ord('5'):
                active_modes[5] = not active_modes[5]
                status = "ON" if active_modes[5] else "OFF"
                print(f"  Face Mesh: {status}")

            elif key == ord('6'):
                active_modes[6] = not active_modes[6]
                status = "ON" if active_modes[6] else "OFF"
                print(f"  Gesture-Controlled PC: {status}")
                if active_modes[6] and not active_modes[3]:
                    active_modes[3] = True
                    print(f"  (Auto-enabled Hand Gesture for control)")

            elif key == ord('7'):
                active_modes[7] = not active_modes[7]
                status = "ON" if active_modes[7] else "OFF"
                print(f"  Drowsiness Detection: {status}")

            elif key == ord('v') or key == ord('V'):
                enabled = ai_features["voice"].toggle()
                print(f"  Voice Feedback: {'ON' if enabled else 'OFF'}")

            elif key == ord('a') or key == ord('A'):
                for k in active_modes:
                    active_modes[k] = True
                print("  All modules: ON")
                print("  (Module 5 auto-enabled to support Module 7 Drowsiness Detection)")

            elif key == ord('n') or key == ord('N'):
                for k in active_modes:
                    active_modes[k] = False
                latency_tracker = LatencyTracker()
                print("  All modules: OFF")

            elif key == ord('r') or key == ord('R'):
                rotation_state = (rotation_state + 1) % 4
                print(f"  Rotation: {ROTATION_LABELS[rotation_state]}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # ── Cleanup ──
        print("Releasing resources...")
        cam.release()
        cv2.destroyAllWindows()
        if modules[3].is_loaded():
            modules[3].release()
        if modules[5].is_loaded():
            modules[5].release()
        ai_features["voice"].shutdown()
        print("Done. Goodbye!")


if __name__ == "__main__":
    main()
