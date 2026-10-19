"""
FIRE_DETECTOR.PY
----------------
This file contains the 'Detector' class which handles the AI vision logic.
It handles:
1. Loading the YOLO AI model from a file.
2. Processing video frames (resizing, color correction).
3. Running the AI model to find 'fire' or 'smoke' in the image.
4. Drawing boxes and labels on the image for visual feedback.
"""
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import logging
from pathlib import Path
from typing import Tuple, Optional, List
from collections import deque


from src.config import Config
from src.detection_status import UNCERTAIN_LABELS, assessment

class Detector:
    def __init__(
        self,
        model_path: Path,
        target_height: int = Config.TARGET_HEIGHT,
        iou_threshold: float = Config.IOU_THRESHOLD,
        min_confidence: float = Config.MIN_CONFIDENCE,
        smoke_confidence: float = Config.SMOKE_CONFIDENCE,
        screen_model_path: Optional[Path] = None,
        device: Optional[str] = None,
        raise_errors: bool = False,
        capture_decisions: bool = False,
        uncertain_labels: bool = False,
        ):
        """
        Initialize the FireDetector with a YOLO model.

        Args:
            model_path (Path): Path to the YOLO model file
            target_height (int): Target height for frame resizing
            iou_threshold (float): IOU threshold for non-maximum suppression
            min_confidence (float): Minimum confidence threshold for detections
        """
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.raise_errors = raise_errors
        self.capture_decisions = capture_decisions
        # Opt-in presentation for the isolated demo. Legacy return strings stay
        # unchanged: existing notification consumers depend on their wording.
        self.uncertain_labels = uncertain_labels
        self.last_result = assessment()
        self.last_decision = None
        self._decision_frame = -1

        try:
            self.model = YOLO(str(model_path))
            self.target_height = target_height
            self.iou_threshold = iou_threshold
            self.min_confidence = min_confidence
            self.smoke_confidence = smoke_confidence
            self.names = self.model.model.names

            # Define colors for different classes
            self.colors = {
                "fire": (0, 0, 255),    # Red for fire
                "smoke": (128, 128, 128)  # Gray for smoke
            }

            # Flame Test Color Chart mapping
            self.FLAME_TEST_COLOR_CHART = {
                "Red": [
                    {"ion": "Lithium", "symbol": "Li+", "color": "Crimson red"},
                    {"ion": "Strontium", "symbol": "Sr2+", "color": "Red"},
                    {"ion": "Cadmium", "symbol": "Cd2+", "color": "Brick red"},
                ],
                "Orange": [
                    {"ion": "Calcium", "symbol": "Ca2+", "color": "Orange-red"},
                ],
                "Yellow": [
                    {"ion": "Sodium", "symbol": "Na+", "color": "Yellow"},
                ],
                "Lilac": [
                    {"ion": "Potassium", "symbol": "K+", "color": "Lilac / pink"},
                ],
                "Green": [
                    {"ion": "Barium", "symbol": "Ba2+", "color": "Green"},
                    {"ion": "Boron", "symbol": "B3+", "color": "Bright green"},
                ],
                "Blue-Green": [
                    {"ion": "Copper", "symbol": "Cu2+", "color": "Blue-green"},
                ],
                "Blue": [
                    {"ion": "Arsenic", "symbol": "As3+", "color": "Blue"},
                    {"ion": "Cesium", "symbol": "Cs+", "color": "Blue-violet"},
                ],
                "White": [
                    {"ion": "Beryllium", "symbol": "Be2+", "color": "White"},
                    {"ion": "Aluminum", "symbol": "Al3+", "color": "Silvery-white"},
                    {"ion": "Cobalt", "symbol": "Co2+", "color": "Silvery white"},
                    {"ion": "Chromium", "symbol": "Cr3+", "color": "Silvery white"},
                    {"ion": "Lead", "symbol": "Pb2+", "color": "Gray-white"},
                ]
            }

            # HSV ranges for flame colors
            self.COLOR_RANGES = {
                "Red": [
                    (np.array([0, 100, 100]), np.array([10, 255, 255])),
                    (np.array([160, 100, 100]), np.array([180, 255, 255]))
                ],
                "Orange": [(np.array([11, 100, 100]), np.array([25, 255, 255]))],
                "Yellow": [(np.array([26, 100, 100]), np.array([35, 255, 255]))],
                "Green": [(np.array([36, 100, 100]), np.array([85, 255, 255]))],
                "Blue-Green": [(np.array([80, 50, 50]), np.array([105, 255, 255]))],
                "Blue": [(np.array([106, 100, 100]), np.array([130, 255, 255]))],
                "Lilac": [(np.array([131, 50, 100]), np.array([165, 255, 255]))],
                "White": [(np.array([0, 0, 200]), np.array([180, 50, 255]))]
            }

            self.logger.info("Fire detector initialized successfully")

            # Anti-Spoofing Configuration
            self.flicker_buffer = deque(maxlen=Config.FLICKER_BUFFER_SIZE)
            self.flicker_threshold = Config.FLICKER_THRESHOLD
            self.last_motion_score = 0.0
            self._decision_history = deque(maxlen=Config.FLICKER_BUFFER_SIZE)

            # Anti-Spoofing: Screen Detection (Standard YOLOv8n)
            self.screen_model = YOLO(str(screen_model_path or "yolov8n.pt"))
            self.screen_classes = [62, 63, 67]  # tv, laptop, cell phone in COCO
        except Exception as e:
            self.logger.error(f"Failed to initialize fire detector: {e}")
            raise

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame maintaining aspect ratio.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Resized frame
        """
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        new_width = int(self.target_height * aspect_ratio)
        return cv2.resize(frame, (new_width, self.target_height))

    def _calculate_rank(self, box: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[int, str, float]:
        """
        Calculate the rank of the detection based on its size relative to the frame.
       \x20
        Args:
            box (np.ndarray): [x1, y1, x2, y2]
            frame_shape (Tuple[int, int]): (height, width)
           \x20
        Returns:
            Tuple[int, str, float]: (rank_id, rank_name, size_percentage)
            Rank 1: Small (< 5%)
            Rank 2: Medium (5% - 20%)
            Rank 3: Large (> 20%)
        """
        x1, y1, x2, y2 = box
        box_area = (x2 - x1) * (y2 - y1)

        height, width = frame_shape
        img_area = height * width

        if img_area == 0:
            return 1, "Small", 0.0

        ratio = box_area / img_area
        percentage = ratio * 100

        if ratio < 0.05:
            return 1, "Small", percentage
        elif ratio < 0.20:
            return 2, "Medium", percentage
        else:
            return 3, "Large", percentage

    def _detect_flame_color(self, roi: np.ndarray) -> Tuple[Optional[str], str]:
        """
        Detect the dominant flame color in the ROI and associated chemical ions.
        """
        if roi.size == 0:
            return None, ""

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color_counts = {}

        for color_name, ranges in self.COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (lower, upper) in ranges:
                m = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, m)

            color_counts[color_name] = cv2.countNonZero(mask)

        dominant_color = max(color_counts, key=color_counts.get)

        # If the dominant color has very few pixels, it might not be a distinct flame color
        if color_counts[dominant_color] < (roi.shape[0] * roi.shape[1] * 0.05):
            return "Natural Fire", "None"

        # Get ions from chart
        ions = self.FLAME_TEST_COLOR_CHART.get(dominant_color, [])
        ion_symbols = [i["symbol"] for i in ions]
        ion_str = ", ".join(ion_symbols) if ion_symbols else "None"

        return dominant_color, ion_str

    def _is_fire_flickering(self, current_roi: np.ndarray) -> bool:
        """
        Check if the fire is flickering by comparing the current ROI with history.
        """
        if current_roi is None or current_roi.size == 0:
            return False

        # Resize to a fixed small size for consistent comparison
        roi_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, (100, 100))

        if len(self.flicker_buffer) < 2:
            self.flicker_buffer.append(roi_resized)
            return True  # Not enough data yet, assume it's real to avoid instant rejection

        diffs = []
        for prev_roi in self.flicker_buffer:
            diff = cv2.absdiff(roi_resized, prev_roi)
            diffs.append(np.mean(diff))

        motion_score = np.mean(diffs)
        self.last_motion_score = motion_score
        self.flicker_buffer.append(roi_resized)

        if motion_score < self.flicker_threshold:
            self.logger.debug(f"Flicker check failed: score {motion_score:.2f} < {self.flicker_threshold}")

        return motion_score > self.flicker_threshold

    def _check_screen_overlap(self, fire_box: np.ndarray, screen_boxes: List[np.ndarray]) -> bool:
        """
        Check if the fire box is inside or significantly overlapping any screen box.
        """
        for s_box in screen_boxes:
            # Calculate intersection
            ix1 = max(fire_box[0], s_box[0])
            iy1 = max(fire_box[1], s_box[1])
            ix2 = min(fire_box[2], s_box[2])
            iy2 = min(fire_box[3], s_box[3])

            if ix2 > ix1 and iy2 > iy1:
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                fire_area = (fire_box[2] - fire_box[0]) * (fire_box[3] - fire_box[1])

                # If > 50% of fire is inside a screen, it's likely fake
                if (intersection_area / fire_area) > 0.5:
                    return True
        return False

    def draw_detection(
        self,
        frame: np.ndarray,
        box: np.ndarray,
        class_name: str,
        confidence: float,
        rank_id: int = 0,
        rank_name: str = "",
        size_pct: float = 0.0,
        custom_color: Optional[Tuple[int, int, int]] = None
    ) -> None:
        """
        Draw a single detection on the frame with enhanced visualization.

        Args:
            frame (np.ndarray): Input frame
            box (np.ndarray): Detection box coordinates [x1, y1, x2, y2]
            class_name (str): Detected class name
            confidence (float): Detection confidence
            rank_id (int): Rank ID (1, 2, 3)
            rank_name (str): Size rank name
            size_pct (float): Size percentage of screen
        """
        x1, y1, x2, y2 = box
        # Default to green if class not found
        color = custom_color if custom_color else self.colors.get(class_name.lower(), (0, 255, 0))

        # Calculate text size for better positioning
        text = f"{class_name}: {confidence:.2f}"
        if rank_id > 0:
            text += f" [Rank {rank_id}: {rank_name} {size_pct:.1f}%]"

        text_scale = 1.5
        if self.uncertain_labels and class_name in UNCERTAIN_LABELS.values():
            # Keep the complete reason and actual score visible on narrow frames.
            text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2)[0][0]
            text_scale *= min(1.0, max(1, frame.shape[1] - max(0, x1) - 10) / max(1, text_width))

        # Add flame specific info if provided in class_name or extra logic
        # For simplicity, we'll let process_frame handle the text formatting if needed,
        # but here we can ensure the label reflects the detailed info.
        label_height = 30  # Approximate height of label
        if y1 < label_height:
            text_y = y2 + label_height  # Place label below box
            rect_y = y2
        else:
            text_y = y1 - 5  # Place label above box
            rect_y = y1

        # Draw semi-transparent background for box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2),
                      color, -1)  # Filled rectangle
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0,
                        frame)  # Transparency effect

        # Draw box outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Add box corners for better visibility
        corner_length = 20
        thickness = 2
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness)

        # Add detection label with enhanced visibility
        cvzone.putTextRect(
            frame,
            text,
            (x1, text_y),
            scale=text_scale,
            thickness=1 if text_scale < 0.7 else 2,
            colorR=color,
            colorT=(255, 255, 255),  # White text
            font=cv2.FONT_HERSHEY_SIMPLEX,
            offset=5,
            border=2,
            colorB=(0, 0, 0),  # Black border
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """
        Process a video frame to detect fire and smoke with enhanced visualization.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            tuple: (processed_frame, detection: str)
        """
        try:
            self._decision_frame += 1
            self.last_result = assessment()
            decision = {"frame_index": self._decision_frame, "candidates": [],
                        "selected_candidate": None, "selected_status": None}
            self.last_decision = decision if self.capture_decisions else None
            # Handle image channels
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            frame = self.resize_frame(frame)
            # Measurements must use camera/video pixels, never earlier box overlays.
            analysis_pixels = frame.copy()
            diagnostic_pixels = analysis_pixels if self.capture_decisions else None
            results = self.model(
                frame, iou=self.iou_threshold, conf=self.min_confidence,
                device=self.device, verbose=False)

            # Anti-Spoofing: Run screen detection
            screen_results = self.screen_model(
                frame, classes=self.screen_classes, conf=0.3,
                device=self.device, verbose=False)
            screen_boxes = []
            if screen_results and len(screen_results[0].boxes) > 0:
                screen_boxes = screen_results[0].boxes.xyxy.cpu().numpy().astype(int)
                self.logger.debug(f"Detected {len(screen_boxes)} screens")

            detection = None
            selected_passes_safeguards = False

            if results:
                result = results[0]

                # Check for Object Detection (Boxes)
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    confidences = result.boxes.conf.cpu().numpy()

                    # Sort detections by confidence
                    sort_idx = np.argsort(-confidences)  # Descending order
                    boxes = boxes[sort_idx]
                    class_ids = class_ids[sort_idx]
                    confidences = confidences[sort_idx]

                    for box, class_id, confidence in zip(boxes, class_ids, confidences):
                        class_name = self.names[class_id]
                        rank_id, rank_name, size_pct = self._calculate_rank(box, frame.shape[:2])

                        # Anti-Spoofing Validation
                        is_fire = "fire" == class_name.lower()
                        is_real = True
                        spoof_reason = ""
                        roi = None
                        history_before = len(self.flicker_buffer)
                        flicker_checked = False
                        motion_score = None
                        prior_annotations = False

                        if is_fire:
                            # 1. Check Screen Overlap
                            if self._check_screen_overlap(box, screen_boxes):
                                is_real = False
                                spoof_reason = "Screen"

                            roi = analysis_pixels[box[1]:box[3], box[0]:box[2]]
                            if self.capture_decisions:
                                raw_roi = diagnostic_pixels[box[1]:box[3], box[0]:box[2]]
                                prior_annotations = not np.array_equal(roi, raw_roi)
                            # 2. Check Flicker (only if not already caught by screen check)
                            if is_real:
                                flicker_checked = True
                                flickering = self._is_fire_flickering(roi)
                                if self.capture_decisions and roi.size:
                                    if history_before >= 2:
                                        motion_score = float(self.last_motion_score)
                                    self._decision_history.append({"frame_index": self._decision_frame,
                                                                   "box": box.tolist()})
                                if not flickering:
                                    is_real = False
                                    spoof_reason = "No Flicker"

                        # Keep confidence order among accepted candidates. A rejected
                        # candidate is only a fallback if no candidate passes checks.
                        previous_detection = detection
                        if detection is None or (not selected_passes_safeguards and is_real):
                            if is_fire and confidence >= self.min_confidence:
                                if is_real:
                                    # Detection of flame color
                                    flame_color, ions = self._detect_flame_color(roi)
                                    color_info = f" [Color: {flame_color}, Ions: {ions}]" if flame_color else ""
                                    detection = f"Fire (Rank {rank_id}){color_info}"
                                    selected_passes_safeguards = True
                                else:
                                    detection = f"Fake Fire ({spoof_reason})"
                            elif "smoke" == class_name.lower() and confidence >= self.smoke_confidence:
                                detection = f"Smoke (Rank {rank_id})"
                                selected_passes_safeguards = True

                        eligible = bool((is_fire and confidence >= self.min_confidence) or
                                        (class_name.lower() == "smoke" and confidence >= self.smoke_confidence))
                        candidate_result = assessment(
                            class_name, eligible=eligible, rejection_reason=spoof_reason or None)
                        if detection != previous_detection:
                            self.last_result = assessment(
                                class_name, eligible=eligible, rejection_reason=spoof_reason or None,
                                legacy_status=detection)

                        if self.capture_decisions:
                            candidate = {"class": class_name, "confidence": float(confidence),
                                         "box": box.tolist(), "passes_safeguards": bool(is_real),
                                         "eligible": eligible,
                                         "assessment": candidate_result,
                                         "rejection_reason": spoof_reason or None,
                                         "flicker_checked": flicker_checked, "motion_score": motion_score,
                                         "flicker_threshold": float(self.flicker_threshold),
                                         "history_size_before": history_before,
                                         "history_after": list(self._decision_history),
                                         "roi_contains_prior_annotations": prior_annotations}
                            decision["candidates"].append(candidate)
                            if detection != previous_detection:
                                decision["selected_candidate"] = len(decision["candidates"]) - 1

                        # Pass the detailed class_name for drawing
                        draw_name = class_name
                        box_color = None

                        if is_fire:
                            if is_real:
                                box_color = self.colors.get("fire")  # Force red for real fire
                                if detection and "Color:" in detection and "Fire" in detection:
                                    # Extract color/ion info for the box label
                                    if "(Rank" in detection:
                                        info = detection.split("(Rank")[1].split(")")[1]
                                        draw_name = f"{class_name}{info}"
                            else:
                                draw_name = (candidate_result["display_label"] if self.uncertain_labels
                                             else f"Fake Fire ({spoof_reason})")
                                box_color = (0, 165, 255)  # Orange for fake fire warning
                        elif "smoke" == class_name.lower():
                            box_color = self.colors.get("smoke")  # Force gray for smoke

                        self.draw_detection(frame, box, draw_name, confidence, rank_id, rank_name, size_pct, custom_color=box_color)

                # Check for Classification (Probs)
                elif hasattr(result, 'probs') and result.probs is not None:
                    probs = result.probs
                    # Get top class
                    top1_index = probs.top1
                    top1_conf = probs.top1conf.item()
                    class_name = self.names[top1_index]

                    if detection is None:
                        if "fire" == class_name.lower() and top1_conf >= self.min_confidence:
                            detection = "Fire"
                        elif "smoke" == class_name.lower() and top1_conf >= self.smoke_confidence:
                            detection = "Smoke"
                        if detection is not None:
                            self.last_result = assessment(class_name, eligible=True, legacy_status=detection)

                    # Draw classification result on screen
                    text = f"Class: {class_name} ({top1_conf:.2f})"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Add frame metadata
            display_status = self.last_result["display_status"] if self.uncertain_labels else detection
            if self.capture_decisions:
                decision["selected_status"] = display_status
                decision["legacy_selected_status"] = detection
                decision["selected_result"] = dict(self.last_result)
                selected = decision["selected_candidate"]
                decision["fake_suppressed_eligible_candidate"] = bool(
                    selected is not None and decision["candidates"][selected]["rejection_reason"] and
                    any(c["eligible"] and c["passes_safeguards"] for c in decision["candidates"]))
            self._add_frame_info(frame, display_status)

            return frame, detection

        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            if self.raise_errors:
                raise
            return frame, None

    def _add_frame_info(self, frame: np.ndarray, detection: Optional[str]) -> None:
        """
        Add frame information overlay.

        Args:
            frame (np.ndarray): Input frame
            detection (Optional[str]): Current detection status
        """
        height, width = frame.shape[:2]

        # Add semi-transparent overlay at the bottom
        overlay_height = 40
        overlay = frame[height-overlay_height:height, 0:width].copy()
        cv2.rectangle(frame, (0, height-overlay_height),
                      (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.2, frame[height-overlay_height:height, 0:width], 0.8, 0,
                        frame[height-overlay_height:height, 0:width])

        # Add status text
        status_text = f"Status: {detection if detection else 'No Detection'}"
        if self.last_motion_score > 0:
            status_text += f" | Motion: {self.last_motion_score:.1f}"

        cv2.putText(frame, status_text, (10, height-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add confidence threshold info
        conf_text = f"Conf: {self.min_confidence:.2f} | IOU: {self.iou_threshold:.2f}"
        text_size = cv2.getTextSize(
            conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(frame, conf_text, (width - text_size[0] - 10, height-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
