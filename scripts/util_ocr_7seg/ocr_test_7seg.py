"""
7-segment wind-speed OCR usage notes.

One-shot read, easiest external API:

    from ocr_7seg import read_wind_speed

    wind = read_wind_speed()
    if wind is None:
        print("OCR failed")
    else:
        print(f"wind={wind:.2f} m/s")

One-shot read with tunable retry:

    wind = read_wind_speed(
        camera_index=1,
        roi_file="ocr_rois.json",
        retry_attempts=5,
        retry_delay_seconds=0.10,
    )

Debug one-shot read:

    wind, info = read_wind_speed(return_debug=True)
    print(info["text"], info["confidence"], info["failure_reason"])
    print(info["retry_candidates"])

Keep the camera open for repeated reads:

    from ocr_7seg import WindSpeedOCR

    with WindSpeedOCR(camera_index=1, roi_file="ocr_rois.json", warmup_frames=30) as reader:
        while True:
            wind, info = reader.recognize_with_retry(
                attempts=5,
                delay_seconds=0.10,
                return_debug=True,
            )
            print(wind, info["text"], info["confidence"])

Keeping WindSpeedOCR alive keeps cv2.VideoCapture open, which gives the camera
time to settle exposure/focus and avoids reopening the device for every reading.
Use recognize_stable_with_retry(...) instead when you want temporal smoothing:

    wind, info = reader.recognize_stable_with_retry(
        attempts=5,
        delay_seconds=0.10,
        return_debug=True,
    )

Before first use, save the display ROI, then drag 4 digit boxes on the
rectified display preview:

    python ocr_7seg/ocr_test_7seg.py --select-roi

To keep the display ROI and only adjust the digit boxes:

    python ocr_7seg/ocr_test_7seg.py --select-digits

For command-line debugging with saved frames/previews:

    python -m ocr_7seg.main_test_ocr_7seg --count 30 --interval 1 --debug-dir debug_ocr_run --raw --preview --retry 5
"""

import json
import os
import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "Seven Segment Wind Speed"
PREVIEW_WINDOW = "Rectified 7-Segment Preview"
ROI_FILE = str(Path(__file__).resolve().parent.parent / "ocr_rois.json")
ROI_KEY = "wind_speed_quad"
DIGIT_BOXES_KEY = "wind_speed_digit_boxes"
ROI_LABEL = "Wind Speed"
ROI_COLOR = (0, 255, 0)
POINT_COLOR = (0, 255, 255)
POLYGON_COLOR = (0, 200, 255)

CAMERA_INDEX = 1  # system camera --> 0; usb camera --> 1

# Perspective-rectified display size.
# Tune these first so the warped display has a clean, front-facing layout.
RECTIFIED_WIDTH = 280
RECTIFIED_HEIGHT = 140

# Fixed digit layout in the rectified coordinate system.
# The meter display is closer to _0.00: one optional leading slot, then
# three visible LCD digits. Keep the boxes on the digit band so labels like
# m/s and C do not get treated as active segments.
DIGIT_BOXES = [
    (0.10, 0.10, 0.27, 0.78),
    (0.27, 0.08, 0.50, 0.64),
    (0.50, 0.08, 0.73, 0.64),
    (0.72, 0.08, 0.97, 0.64),
]
DECIMAL_AFTER_DIGIT = 2
OPTIONAL_LEADING_DIGITS = 1
USE_SLOT_LOCAL_DIGIT_BOXES = True
USE_DYNAMIC_DIGIT_BOXES = False
DIGIT_SEARCH_REGION = (0.04, 0.08, 0.96, 0.64)
MIN_DYNAMIC_DIGIT_WIDTH = 8
MIN_DYNAMIC_DIGIT_HEIGHT = 28
MIN_DYNAMIC_COLUMN_PIXELS = 6
MAX_DYNAMIC_GAP = 10
MIN_SLOT_PIXEL_COUNT = 20

UPSCALE = 3
MAX_REASONABLE_SPEED = 80.0
BLANK_ACTIVE_RATIO = 0.010
LEADING_ONE_RAW_ACTIVE_RATIO = 0.080
MAX_LEADING_ONE_BOX_WIDTH_RATIO = 0.120
NARROW_ONE_MAX_WIDTH_HEIGHT_RATIO = 0.42
NARROW_ONE_MAX_WIDTH_RATIO = 0.115
NARROW_ONE_MIN_RAW_ACTIVE_RATIO = 0.12
OPTIONAL_LEADING_DROP_CONFIDENCE = 0.70

HORIZONTAL_SEGMENTS = {0, 3, 6}
VERTICAL_SEGMENTS = {1, 2, 4, 5}
SEGMENT_ON_THRESHOLD_H = 0.34
SEGMENT_ON_THRESHOLD_V = 0.25
SEGMENT_OFF_THRESHOLD_H = 0.20
SEGMENT_OFF_THRESHOLD_V = 0.13

RESULT_HISTORY_SIZE = 12
STABLE_SCORE_THRESHOLD = 5.5
SWITCH_MARGIN = 1.2
RETRY_GOOD_CONFIDENCE = 0.65

# Segment order:
# top, upper_left, upper_right, middle, lower_left, lower_right, bottom
SEGMENT_DIGITS = {
    (1, 1, 1, 0, 1, 1, 1): "0",
    (0, 0, 1, 0, 0, 1, 0): "1",
    (1, 0, 1, 1, 1, 0, 1): "2",
    (1, 0, 1, 1, 0, 1, 1): "3",
    (0, 1, 1, 1, 0, 1, 0): "4",
    (1, 1, 0, 1, 0, 1, 1): "5",
    (1, 1, 0, 1, 1, 1, 1): "6",
    (1, 0, 1, 0, 0, 1, 0): "7",
    (1, 1, 1, 1, 1, 1, 1): "8",
    (1, 1, 1, 1, 0, 1, 1): "9",
}

DIGIT_SEGMENTS = {digit: state for state, digit in SEGMENT_DIGITS.items()}


def clamp_point(x, y, width, height):
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    return x, y


def order_quad_points(points):
    pts = np.array(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Need exactly 4 points.")

    sums = pts.sum(axis=1)
    diffs = pts[:, 0] - pts[:, 1]

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]   # top-left
    ordered[2] = pts[np.argmax(sums)]   # bottom-right
    ordered[1] = pts[np.argmax(diffs)]  # top-right
    ordered[3] = pts[np.argmin(diffs)]  # bottom-left
    return ordered


def is_valid_quad(points):
    try:
        ordered = order_quad_points(points)
    except ValueError:
        return False

    contour = ordered.astype(np.float32)
    area = abs(cv2.contourArea(contour))
    return area > 100.0


def read_roi_data(roi_file=ROI_FILE):
    if not os.path.exists(roi_file):
        return {}

    with open(roi_file, "r", encoding="utf-8") as file:
        return json.load(file)


def write_roi_data(data, roi_file=ROI_FILE):
    with open(roi_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_quad(points, roi_file=ROI_FILE):
    data = read_roi_data(roi_file)
    ordered = order_quad_points(points)
    data[ROI_KEY] = [[int(x), int(y)] for x, y in ordered]
    write_roi_data(data, roi_file)


def load_quad(roi_file=ROI_FILE):
    data = read_roi_data(roi_file)
    value = data.get(ROI_KEY)
    if not isinstance(value, list) or len(value) != 4:
        return None

    points = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not all(isinstance(coord, int) for coord in item)
        ):
            return None
        points.append((item[0], item[1]))

    if not is_valid_quad(points):
        return None

    return [tuple(point) for point in order_quad_points(points).astype(int)]


def clear_saved_quad(roi_file=ROI_FILE):
    data = read_roi_data(roi_file)
    data.pop(ROI_KEY, None)
    data.pop(DIGIT_BOXES_KEY, None)
    write_roi_data(data, roi_file)


def is_valid_digit_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False

    try:
        x1, y1, x2, y2 = [float(value) for value in box]
    except (TypeError, ValueError):
        return False

    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


def save_digit_boxes(digit_boxes, roi_file=ROI_FILE):
    if len(digit_boxes) != len(DIGIT_BOXES):
        raise ValueError(f"Need exactly {len(DIGIT_BOXES)} digit boxes.")

    normalized = []
    for box in digit_boxes:
        if not is_valid_digit_box(box):
            raise ValueError("Digit boxes must be normalized x1,y1,x2,y2 values.")
        normalized.append([round(float(value), 6) for value in box])

    data = read_roi_data(roi_file)
    data[DIGIT_BOXES_KEY] = normalized
    write_roi_data(data, roi_file)


def load_digit_boxes(roi_file=ROI_FILE):
    data = read_roi_data(roi_file)
    value = data.get(DIGIT_BOXES_KEY)
    if not isinstance(value, list) or len(value) != len(DIGIT_BOXES):
        return None

    boxes = []
    for box in value:
        if not is_valid_digit_box(box):
            return None
        boxes.append(tuple(float(item) for item in box))

    return sorted(boxes, key=lambda box: box[0])


def clear_saved_digit_boxes(roi_file=ROI_FILE):
    data = read_roi_data(roi_file)
    data.pop(DIGIT_BOXES_KEY, None)
    write_roi_data(data, roi_file)


def warp_quad(frame, points):
    src = order_quad_points(points)
    dst = np.array(
        [
            [0, 0],
            [RECTIFIED_WIDTH - 1, 0],
            [RECTIFIED_WIDTH - 1, RECTIFIED_HEIGHT - 1],
            [0, RECTIFIED_HEIGHT - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        frame,
        matrix,
        (RECTIFIED_WIDTH, RECTIFIED_HEIGHT),
        flags=cv2.INTER_CUBIC,
    )
    return warped


def segment_regions():
    return [
        ((0.22, 0.04), (0.78, 0.17)),
        ((0.07, 0.17), (0.26, 0.46)),
        ((0.74, 0.17), (0.93, 0.46)),
        ((0.22, 0.43), (0.78, 0.58)),
        ((0.07, 0.54), (0.26, 0.83)),
        ((0.74, 0.54), (0.93, 0.83)),
        ((0.22, 0.84), (0.78, 0.97)),
    ]


def build_segment_template(state, size=(72, 120)):
    template = np.zeros((size[1], size[0]), dtype=np.uint8)
    for is_on, ((x1r, y1r), (x2r, y2r)) in zip(state, segment_regions()):
        if not is_on:
            continue
        x1 = int(size[0] * x1r)
        y1 = int(size[1] * y1r)
        x2 = int(size[0] * x2r)
        y2 = int(size[1] * y2r)
        cv2.rectangle(template, (x1, y1), (x2, y2), 255, -1)

    template = cv2.GaussianBlur(template, (5, 5), 0)
    _, template = cv2.threshold(template, 10, 255, cv2.THRESH_BINARY)
    return template


DIGIT_TEMPLATES = {
    digit: build_segment_template(state) for state, digit in SEGMENT_DIGITS.items()
}


def preprocess_rectified(rectified_bgr):
    gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    blackhat = cv2.morphologyEx(
        contrast,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
    )
    enhanced = cv2.addWeighted(contrast, 1.0, blackhat, 1.25, 0)

    _, binary = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=2,
    )
    binary = cv2.medianBlur(binary, 3)
    return gray, enhanced, binary


def digit_box_ratios_to_px(binary_rectified, digit_boxes):
    height, width = binary_rectified.shape[:2]
    boxes = []

    for x1r, y1r, x2r, y2r in digit_boxes:
        x1 = max(0, int(width * x1r))
        y1 = max(0, int(height * y1r))
        x2 = min(width, int(width * x2r))
        y2 = min(height, int(height * y2r))
        boxes.append((x1, y1, x2, y2))

    return boxes


def fixed_digit_boxes_px(binary_rectified):
    return digit_box_ratios_to_px(binary_rectified, DIGIT_BOXES)


def slot_local_digit_boxes_px(binary_rectified, digit_boxes=None):
    boxes = []
    height, width = binary_rectified.shape[:2]
    digit_boxes = digit_boxes if digit_boxes is not None else DIGIT_BOXES

    for x1r, y1r, x2r, y2r in digit_boxes:
        sx1 = max(0, int(width * x1r))
        sy1 = max(0, int(height * y1r))
        sx2 = min(width, int(width * x2r))
        sy2 = min(height, int(height * y2r))
        slot = binary_rectified[sy1:sy2, sx1:sx2]

        ys, xs = np.where(slot > 0)
        if len(xs) < MIN_SLOT_PIXEL_COUNT or len(ys) < MIN_SLOT_PIXEL_COUNT:
            boxes.append((sx1, sy1, sx2, sy2))
            continue

        x1 = max(sx1, sx1 + int(xs.min()) - 2)
        y1 = max(sy1, sy1 + int(ys.min()) - 2)
        x2 = min(sx2, sx1 + int(xs.max()) + 3)
        y2 = min(sy2, sy1 + int(ys.max()) + 3)

        if x2 - x1 < MIN_DYNAMIC_DIGIT_WIDTH or y2 - y1 < MIN_DYNAMIC_DIGIT_HEIGHT:
            boxes.append((sx1, sy1, sx2, sy2))
        else:
            boxes.append((x1, y1, x2, y2))

    return boxes


def merge_close_ranges(ranges, max_gap):
    if not ranges:
        return []

    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged


def dynamic_digit_boxes_px(binary_rectified):
    height, width = binary_rectified.shape[:2]
    x1r, y1r, x2r, y2r = DIGIT_SEARCH_REGION
    sx1 = max(0, int(width * x1r))
    sy1 = max(0, int(height * y1r))
    sx2 = min(width, int(width * x2r))
    sy2 = min(height, int(height * y2r))
    search = binary_rectified[sy1:sy2, sx1:sx2]

    if search.size == 0:
        return []

    column_counts = cv2.countNonZero(search) if search.ndim == 1 else np.count_nonzero(search, axis=0)
    active_columns = column_counts >= MIN_DYNAMIC_COLUMN_PIXELS
    ranges = []
    start = None

    for index, is_active in enumerate(active_columns):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            ranges.append((start, index))
            start = None

    if start is not None:
        ranges.append((start, len(active_columns)))

    ranges = merge_close_ranges(ranges, MAX_DYNAMIC_GAP)
    boxes = []

    for start, end in ranges:
        if end - start < MIN_DYNAMIC_DIGIT_WIDTH:
            continue

        candidate = search[:, start:end]
        ys, xs = np.where(candidate > 0)
        if len(xs) < 20 or len(ys) < 20:
            continue

        x1 = sx1 + start + int(xs.min()) - 2
        x2 = sx1 + start + int(xs.max()) + 3
        y1 = sy1 + int(ys.min()) - 2
        y2 = sy1 + int(ys.max()) + 3

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 - x1 < MIN_DYNAMIC_DIGIT_WIDTH or y2 - y1 < MIN_DYNAMIC_DIGIT_HEIGHT:
            continue

        boxes.append((x1, y1, x2, y2))

    boxes = sorted(boxes, key=lambda box: box[0])
    max_digits = len(DIGIT_BOXES)

    if len(boxes) > max_digits:
        boxes = sorted(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
        boxes = sorted(boxes[:max_digits], key=lambda box: box[0])

    if len(boxes) < max_digits - OPTIONAL_LEADING_DIGITS:
        return []

    fixed_boxes = fixed_digit_boxes_px(binary_rectified)
    aligned_boxes = fixed_boxes[:]
    start_slot = max_digits - len(boxes)

    for index, box in enumerate(boxes):
        aligned_boxes[start_slot + index] = box

    return aligned_boxes


def extract_digit_rois(binary_rectified, manual_digit_boxes=None):
    if manual_digit_boxes is not None:
        digit_boxes_px = (
            slot_local_digit_boxes_px(binary_rectified, manual_digit_boxes)
            if USE_SLOT_LOCAL_DIGIT_BOXES
            else digit_box_ratios_to_px(binary_rectified, manual_digit_boxes)
        )
    else:
        digit_boxes_px = slot_local_digit_boxes_px(binary_rectified) if USE_SLOT_LOCAL_DIGIT_BOXES else []
    if not digit_boxes_px:
        digit_boxes_px = dynamic_digit_boxes_px(binary_rectified) if USE_DYNAMIC_DIGIT_BOXES else []
    if not digit_boxes_px:
        digit_boxes_px = fixed_digit_boxes_px(binary_rectified)

    digit_images = [
        binary_rectified[y1:y2, x1:x2].copy()
        for x1, y1, x2, y2 in digit_boxes_px
    ]

    return digit_images, digit_boxes_px


def tighten_digit_crop(digit_binary):
    if digit_binary.size == 0:
        return digit_binary

    ys, xs = np.where(digit_binary > 0)
    if len(xs) < 15 or len(ys) < 15:
        return digit_binary

    x1 = max(0, int(xs.min()) - 2)
    y1 = max(0, int(ys.min()) - 2)
    x2 = min(digit_binary.shape[1], int(xs.max()) + 3)
    y2 = min(digit_binary.shape[0], int(ys.max()) + 3)
    return digit_binary[y1:y2, x1:x2]


def normalize_digit_binary(digit_binary):
    if digit_binary.size == 0:
        return np.zeros((120, 72), dtype=np.uint8)

    digit_binary = tighten_digit_crop(digit_binary)
    pad_y = max(2, digit_binary.shape[0] // 20)
    pad_x = max(2, digit_binary.shape[1] // 16)
    digit_binary = cv2.copyMakeBorder(
        digit_binary,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    digit_binary = cv2.resize(digit_binary, (72, 120), interpolation=cv2.INTER_NEAREST)
    digit_binary = cv2.morphologyEx(
        digit_binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    return digit_binary


def digit_active_ratio(digit_binary):
    if digit_binary.size == 0:
        return 0.0
    return cv2.countNonZero(digit_binary) / float(digit_binary.size)


def segment_fill_scores(digit_binary):
    norm = normalize_digit_binary(digit_binary)
    scores = []

    for index, ((x1r, y1r), (x2r, y2r)) in enumerate(segment_regions()):
        x1 = max(0, int(norm.shape[1] * x1r))
        y1 = max(0, int(norm.shape[0] * y1r))
        x2 = min(norm.shape[1], int(norm.shape[1] * x2r))
        y2 = min(norm.shape[0], int(norm.shape[0] * y2r))

        patch = norm[y1:y2, x1:x2]
        if patch.size == 0:
            scores.append(0.0)
            continue

        fill_ratio = cv2.countNonZero(patch) / float(patch.size)
        if index in HORIZONTAL_SEGMENTS:
            center_band = patch[patch.shape[0] // 3 : (patch.shape[0] * 2) // 3, :]
        else:
            center_band = patch[:, patch.shape[1] // 3 : (patch.shape[1] * 2) // 3]
        center_ratio = (
            cv2.countNonZero(center_band) / float(center_band.size)
            if center_band.size
            else 0.0
        )

        scores.append(0.55 * fill_ratio + 0.45 * center_ratio)

    return norm, scores


def decode_segments_from_scores(scores):
    state = []
    confidence = 0.0

    for index, score in enumerate(scores):
        if index in HORIZONTAL_SEGMENTS:
            on_threshold = SEGMENT_ON_THRESHOLD_H
            off_threshold = SEGMENT_OFF_THRESHOLD_H
        else:
            on_threshold = SEGMENT_ON_THRESHOLD_V
            off_threshold = SEGMENT_OFF_THRESHOLD_V

        threshold = (on_threshold + off_threshold) / 2.0
        margin = max(abs(score - threshold), 0.02)
        confidence += min(margin * 6.0, 1.0)

        if score >= on_threshold:
            state.append(1)
        elif score <= off_threshold:
            state.append(0)
        else:
            state.append(1 if score >= threshold else 0)

    return tuple(state), confidence / 7.0


def template_match_score(norm_digit_binary, template):
    intersection = np.logical_and(norm_digit_binary > 0, template > 0).sum()
    union = np.logical_or(norm_digit_binary > 0, template > 0).sum()
    if union == 0:
        return 0.0

    overlap = intersection / float(union)
    similarity = cv2.matchTemplate(norm_digit_binary, template, cv2.TM_CCOEFF_NORMED)[0][0]
    similarity = max(0.0, float(similarity))
    return 0.65 * overlap + 0.35 * similarity


def segment_distance(state_a, state_b):
    return sum(1 for left, right in zip(state_a, state_b) if left != right)


def decode_digit_hybrid(digit_binary):
    raw_active_ratio = digit_active_ratio(digit_binary)
    norm, scores = segment_fill_scores(digit_binary)
    active_ratio = digit_active_ratio(norm)
    if raw_active_ratio < BLANK_ACTIVE_RATIO:
        return {
            "digit": "",
            "confidence": 0.95,
            "state": tuple(0 for _ in range(7)),
            "scores": scores,
            "binary": norm,
            "template_digit": "",
            "template_score": 0.0,
            "active_ratio": active_ratio,
            "raw_active_ratio": raw_active_ratio,
            "is_blank": True,
        }

    state, state_confidence = decode_segments_from_scores(scores)
    segment_digit = SEGMENT_DIGITS.get(state, "")

    best_template_digit = ""
    best_template_score = -1.0
    second_template_score = -1.0
    template_scores = {}

    for digit, template in DIGIT_TEMPLATES.items():
        score = template_match_score(norm, template)
        template_scores[digit] = score
        if score > best_template_score:
            second_template_score = best_template_score
            best_template_score = score
            best_template_digit = digit
        elif score > second_template_score:
            second_template_score = score

    winner = best_template_digit
    confidence = best_template_score

    if segment_digit:
        segment_template_score = template_scores.get(segment_digit, 0.0)
        if segment_digit == best_template_digit:
            winner = segment_digit
            confidence = max(best_template_score, 0.75 + 0.20 * state_confidence)
        elif segment_template_score >= best_template_score - 0.06:
            winner = segment_digit
            confidence = 0.55 * segment_template_score + 0.45 * state_confidence
    else:
        best_template_state = DIGIT_SEGMENTS.get(best_template_digit)
        distance = segment_distance(state, best_template_state) if best_template_state else 7
        if best_template_score >= 0.30 and distance <= 1:
            winner = best_template_digit
            confidence = max(best_template_score, 0.62)
        elif best_template_score >= 0.38 and distance <= 2:
            winner = best_template_digit
            confidence = max(best_template_score, 0.55)

    confidence_gap = best_template_score - max(second_template_score, 0.0)
    confidence = min(1.0, confidence + max(0.0, confidence_gap) * 0.35)

    if confidence < 0.40:
        winner = ""

    return {
        "digit": winner,
        "confidence": confidence,
        "state": state,
        "scores": scores,
        "binary": norm,
        "template_digit": best_template_digit,
        "template_score": best_template_score,
        "active_ratio": active_ratio,
        "raw_active_ratio": raw_active_ratio,
        "is_blank": False,
    }


def coerce_optional_leading_digits(digit_results, digit_boxes_px, image_width):
    for index in range(min(OPTIONAL_LEADING_DIGITS, len(digit_results))):
        result = digit_results[index]
        if result["digit"] or result["is_blank"]:
            continue
        if result.get("raw_active_ratio", 0.0) < LEADING_ONE_RAW_ACTIVE_RATIO:
            continue
        x1, y1, x2, y2 = digit_boxes_px[index]
        if (x2 - x1) / float(max(image_width, 1)) > MAX_LEADING_ONE_BOX_WIDTH_RATIO:
            continue

        result["digit"] = "1"
        result["confidence"] = max(result["confidence"], 0.70)
        result["coerced"] = "optional-leading-1"

    return digit_results


def coerce_narrow_one_digits(digit_results, digit_boxes_px, image_width):
    for result, (x1, y1, x2, y2) in zip(digit_results, digit_boxes_px):
        if result["digit"] or result["is_blank"]:
            continue

        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        if width / float(height) > NARROW_ONE_MAX_WIDTH_HEIGHT_RATIO:
            continue
        if width / float(max(image_width, 1)) > NARROW_ONE_MAX_WIDTH_RATIO:
            continue
        if result.get("raw_active_ratio", 0.0) < NARROW_ONE_MIN_RAW_ACTIVE_RATIO:
            continue

        result["digit"] = "1"
        result["confidence"] = max(result["confidence"], 0.70)
        result["coerced"] = "narrow-1"

    return digit_results


def assemble_number(digit_results):
    digits = []
    total_confidence = 0.0

    for index, result in enumerate(digit_results):
        digit = result["digit"]
        is_optional_leading = index < OPTIONAL_LEADING_DIGITS

        if result["is_blank"] and is_optional_leading:
            digits.append("")
        else:
            digits.append(digit)

        total_confidence += result["confidence"]

    if any(not digit for digit in digits[OPTIONAL_LEADING_DIGITS:]):
        missing = [
            f"D{index + 1}"
            for index, digit in enumerate(digits[OPTIONAL_LEADING_DIGITS:], start=OPTIONAL_LEADING_DIGITS)
            if not digit
        ]
        return "", total_confidence / max(len(digit_results), 1), f"missing_required_digits:{','.join(missing)}"

    physical_chars = []
    for index, digit in enumerate(digits):
        physical_chars.append(digit)
        if index + 1 == DECIMAL_AFTER_DIGIT:
            physical_chars.append(".")

    text = "".join(physical_chars).strip()

    for index in range(OPTIONAL_LEADING_DIGITS):
        if not text:
            break
        if digits[index] == "":
            continue
        if digits[index] != "0":
            break
        text = text[1:]

    if text.startswith("."):
        text = "0" + text

    if "." in text:
        integer_part, decimal_part = text.split(".", 1)
        integer_part = integer_part.lstrip("0") or "0"
        text = f"{integer_part}.{decimal_part}"

    return text, total_confidence / max(len(digit_results), 1), "ok"


def copy_digit_result(result):
    return dict(result)


def drop_optional_leading_digits(digit_results):
    adjusted = [copy_digit_result(result) for result in digit_results]

    for index in range(min(OPTIONAL_LEADING_DIGITS, len(adjusted))):
        adjusted[index]["digit"] = ""
        adjusted[index]["confidence"] = max(
            adjusted[index].get("confidence", 0.0),
            OPTIONAL_LEADING_DROP_CONFIDENCE,
        )
        adjusted[index]["is_blank"] = True
        adjusted[index]["coerced"] = "dropped-optional-leading"

    return adjusted


def is_reasonable_speed(text):
    if not text:
        return False

    try:
        value = float(text)
    except ValueError:
        return False

    return 0.0 <= value <= MAX_REASONABLE_SPEED


def make_digit_panel(index, result):
    panel = cv2.cvtColor(result["binary"], cv2.COLOR_GRAY2BGR)
    digit_text = "blank" if result["is_blank"] else (result["digit"] if result["digit"] else "?")
    score_text = " ".join(f"{value:.2f}" for value in result["scores"])

    cv2.putText(panel, f"D{index + 1}: {digit_text}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0) if result["digit"] or result["is_blank"] else (0, 0, 255), 1)
    cv2.putText(panel, "".join(str(bit) for bit in result["state"]), (4, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    cv2.putText(panel, f"T:{result['template_digit']} {result['template_score']:.2f}", (4, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)
    cv2.putText(panel, f"C:{result['confidence']:.2f}", (4, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.putText(panel, f"A:{result['active_ratio']:.3f}", (4, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 255, 180), 1)
    cv2.putText(panel, score_text, (4, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (160, 160, 160), 1)
    return panel


def pad_images_to_same_height(images):
    if not images:
        return images

    max_height = max(image.shape[0] for image in images)
    padded = []
    for image in images:
        height = image.shape[0]
        if height == max_height:
            padded.append(image)
            continue
        padded.append(
            cv2.copyMakeBorder(
                image,
                0,
                max_height - height,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        )
    return padded


def pad_images_to_same_width(images):
    if not images:
        return images

    max_width = max(image.shape[1] for image in images)
    padded = []
    for image in images:
        width = image.shape[1]
        if width == max_width:
            padded.append(image)
            continue
        padded.append(
            cv2.copyMakeBorder(
                image,
                0,
                0,
                0,
                max_width - width,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        )
    return padded


def build_debug_preview(rectified_bgr, rectified_gray, rectified_enhanced, rectified_binary,
                        digit_boxes_px, digit_results, raw_text):
    warped_view = rectified_bgr.copy()
    binary_view = cv2.cvtColor(rectified_binary, cv2.COLOR_GRAY2BGR)

    for index, (x1, y1, x2, y2) in enumerate(digit_boxes_px):
        cv2.rectangle(warped_view, (x1, y1), (x2, y2), ROI_COLOR, 1)
        cv2.rectangle(binary_view, (x1, y1), (x2, y2), ROI_COLOR, 1)
        cv2.putText(warped_view, f"D{index + 1}", (x1 + 2, max(16, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, POINT_COLOR, 1)

    if 0 < DECIMAL_AFTER_DIGIT < len(DIGIT_BOXES):
        decimal_x = digit_boxes_px[DECIMAL_AFTER_DIGIT - 1][2]
        cv2.line(binary_view, (decimal_x, 0), (decimal_x, binary_view.shape[0] - 1), (255, 255, 0), 1)

    gray_view = cv2.cvtColor(rectified_gray, cv2.COLOR_GRAY2BGR)
    enhanced_view = cv2.cvtColor(rectified_enhanced, cv2.COLOR_GRAY2BGR)
    top_row = cv2.hconcat(
        pad_images_to_same_height([warped_view, gray_view, enhanced_view, binary_view])
    )

    digit_panels = [make_digit_panel(index, result) for index, result in enumerate(digit_results)]
    digit_row = cv2.hconcat(pad_images_to_same_height(digit_panels))

    top_row, digit_row = pad_images_to_same_width([top_row, digit_row])
    preview = cv2.vconcat([top_row, digit_row])
    cv2.putText(
        preview,
        f"raw: {raw_text if raw_text else '(empty)'}",
        (10, preview.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )
    return preview


def summarize_digit_results(digit_results, digit_boxes_px):
    parts = []
    for index, (result, box) in enumerate(zip(digit_results, digit_boxes_px)):
        digit = "blank" if result["is_blank"] else (result["digit"] if result["digit"] else "?")
        state = "".join(str(bit) for bit in result["state"])
        scores = "/".join(f"{score:.2f}" for score in result["scores"])
        box_text = ",".join(str(value) for value in box)
        coerced = result.get("coerced", "")
        parts.append(
            f"D{index + 1}:{digit};state={state};tmpl={result['template_digit']}"
            f";ts={result['template_score']:.2f};conf={result['confidence']:.2f}"
            f";active={result['active_ratio']:.3f};coerced={coerced}"
            f";raw_active={result.get('raw_active_ratio', 0.0):.3f}"
            f";box={box_text};scores={scores}"
        )
    return " | ".join(parts)


def summarize_digit_boxes(digit_boxes_px):
    parts = []
    for index, (x1, y1, x2, y2) in enumerate(digit_boxes_px):
        parts.append(f"D{index + 1}:{x1},{y1},{x2},{y2},w={x2 - x1},h={y2 - y1}")
    return " | ".join(parts)


def recognize_wind_speed(frame, quad_points, return_debug=False, manual_digit_boxes=None):
    rectified = warp_quad(frame, quad_points)
    rectified_gray, rectified_enhanced, rectified_binary = preprocess_rectified(rectified)
    digit_images, digit_boxes_px = extract_digit_rois(rectified_binary, manual_digit_boxes)

    digit_results = [decode_digit_hybrid(digit_image) for digit_image in digit_images]
    digit_results = coerce_narrow_one_digits(
        digit_results,
        digit_boxes_px,
        rectified_binary.shape[1],
    )
    digit_results = coerce_optional_leading_digits(
        digit_results,
        digit_boxes_px,
        rectified_binary.shape[1],
    )
    text, confidence, failure_reason = assemble_number(digit_results)

    if not is_reasonable_speed(text):
        original_text = text
        if original_text and OPTIONAL_LEADING_DIGITS > 0:
            adjusted_digit_results = drop_optional_leading_digits(digit_results)
            adjusted_text, adjusted_confidence, adjusted_reason = assemble_number(
                adjusted_digit_results
            )
            if is_reasonable_speed(adjusted_text):
                digit_results = adjusted_digit_results
                text = adjusted_text
                confidence = min(confidence, adjusted_confidence)
                failure_reason = f"dropped_optional_leading:{original_text}"

    if not is_reasonable_speed(text):
        if text:
            failure_reason = f"unreasonable_speed:{text}"
        text = ""
        confidence *= 0.5

    preview = None
    debug_info = None
    if return_debug:
        preview = build_debug_preview(
            rectified,
            rectified_gray,
            rectified_enhanced,
            rectified_binary,
            digit_boxes_px,
            digit_results,
            text,
        )
        debug_info = {
            "preview": preview,
            "digit_summary": summarize_digit_results(digit_results, digit_boxes_px),
            "digit_box_summary": summarize_digit_boxes(digit_boxes_px),
            "digit_boxes": digit_boxes_px,
            "manual_digit_boxes": manual_digit_boxes,
            "digit_results": digit_results,
            "digit_images": digit_images,
            "rectified": rectified,
            "rectified_gray": rectified_gray,
            "rectified_enhanced": rectified_enhanced,
            "rectified_binary": rectified_binary,
            "failure_reason": failure_reason,
        }
    return text, confidence, debug_info


class ResultStabilizer:
    def __init__(self, maxlen=RESULT_HISTORY_SIZE):
        self.history = deque(maxlen=maxlen)
        self.stable_text = ""

    def reset(self):
        self.history.clear()
        self.stable_text = ""

    def update(self, candidate, confidence):
        if candidate:
            self.history.append((candidate, max(0.0, min(1.0, confidence))))
        else:
            self.history.append(("", 0.0))

        scores = {}
        latest_index = {}
        for index, (text, score) in enumerate(self.history):
            if not text:
                continue
            scores[text] = scores.get(text, 0.0) + score
            latest_index[text] = index

        if not scores:
            return self.stable_text

        ordered = sorted(
            scores.items(),
            key=lambda item: (item[1], latest_index[item[0]]),
            reverse=True,
        )
        best_text, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0

        if not self.stable_text:
            if best_score >= STABLE_SCORE_THRESHOLD or (
                candidate == best_text and confidence >= 0.82
            ):
                self.stable_text = best_text
            return self.stable_text

        current_score = scores.get(self.stable_text, 0.0)
        if best_text == self.stable_text:
            return self.stable_text

        if best_score >= max(current_score + SWITCH_MARGIN, second_score + SWITCH_MARGIN):
            self.stable_text = best_text
            return self.stable_text

        if candidate == self.stable_text and confidence >= 0.72:
            return self.stable_text

        return self.stable_text


def draw_quad_overlay(frame, points, closed=True, point_numbers=True):
    if not points:
        return

    for index, (x, y) in enumerate(points):
        cv2.circle(frame, (x, y), 5, POINT_COLOR, -1)
        if point_numbers:
            cv2.putText(frame, str(index + 1), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, POINT_COLOR, 2)

    if len(points) >= 2:
        poly = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [poly], closed and len(points) == 4, POLYGON_COLOR, 2)


def draw_digit_boxes_overlay(frame, boxes, active_box=None):
    height, width = frame.shape[:2]

    for index, (x1r, y1r, x2r, y2r) in enumerate(boxes):
        x1, y1 = int(width * x1r), int(height * y1r)
        x2, y2 = int(width * x2r), int(height * y2r)
        cv2.rectangle(frame, (x1, y1), (x2, y2), ROI_COLOR, 2)
        cv2.putText(
            frame,
            f"D{index + 1}",
            (x1 + 4, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            ROI_COLOR,
            1,
        )

    if active_box is not None:
        x1, y1, x2, y2 = active_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), POINT_COLOR, 1)


def capture_frame(camera_index=CAMERA_INDEX, warmup_frames=8):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera_index}.")

    try:
        frame = None
        for _ in range(max(1, warmup_frames)):
            ok, frame = cap.read()
            if not ok:
                frame = None

        if frame is None:
            raise RuntimeError("Failed to read frame from camera.")
        return frame
    finally:
        cap.release()


def text_to_float(text):
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


class WindSpeedOCR:
    """
    Keep the camera open and read wind speed on demand.

    Use this in the main project when repeated measurements are needed. Keeping
    VideoCapture alive gives the camera time to settle focus and exposure, and
    avoids reopening the device for every reading.
    """

    def __init__(
        self,
        camera_index=CAMERA_INDEX,
        roi_file=ROI_FILE,
        quad_points=None,
        warmup_frames=20,
        buffer_size=1,
    ):
        self.camera_index = camera_index
        self.roi_file = roi_file
        self.quad_points = quad_points if quad_points is not None else load_quad(roi_file)
        self.manual_digit_boxes = load_digit_boxes(roi_file)
        if self.quad_points is None:
            raise RuntimeError("No saved ROI. Run this file with --select-roi first.")

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_index}.")

        if buffer_size is not None:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

        self.stabilizer = ResultStabilizer()
        self.warmup(warmup_frames)

    def warmup(self, frame_count=20):
        for _ in range(max(0, frame_count)):
            self.cap.read()

    def capture_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from camera.")
        return frame

    def _recognition_info(self, frame, text, confidence, debug_info):
        return {
            "text": text,
            "confidence": confidence,
            "quad_points": self.quad_points,
            "frame": frame,
            "preview": debug_info["preview"],
            "digit_summary": debug_info["digit_summary"],
            "digit_box_summary": debug_info["digit_box_summary"],
            "digit_boxes": debug_info["digit_boxes"],
            "digit_images": debug_info["digit_images"],
            "rectified": debug_info["rectified"],
            "rectified_gray": debug_info["rectified_gray"],
            "rectified_enhanced": debug_info["rectified_enhanced"],
            "rectified_binary": debug_info["rectified_binary"],
            "failure_reason": debug_info["failure_reason"],
        }

    def _recognize_debug_frame(self):
        frame = self.capture_frame()
        text, confidence, debug_info = recognize_wind_speed(
            frame,
            self.quad_points,
            return_debug=True,
            manual_digit_boxes=self.manual_digit_boxes,
        )
        return frame, text, confidence, debug_info

    def _retry_candidate_score(self, text, confidence, debug_info):
        if not text or not is_reasonable_speed(text):
            return max(0.0, confidence) * 0.1

        failure_reason = debug_info.get("failure_reason", "")
        reason_bonus = 0.0
        if failure_reason == "ok":
            reason_bonus = 1.0
        elif failure_reason.startswith("dropped_optional_leading:"):
            reason_bonus = 0.4

        return 10.0 + reason_bonus + max(0.0, confidence)

    def recognize_with_retry(
        self,
        attempts=3,
        delay_seconds=0.10,
        return_debug=False,
        good_confidence=RETRY_GOOD_CONFIDENCE,
    ):
        attempts = max(1, int(attempts))
        delay_seconds = max(0.0, float(delay_seconds))

        best = None
        candidates = []
        for attempt_index in range(1, attempts + 1):
            frame, text, confidence, debug_info = self._recognize_debug_frame()
            score = self._retry_candidate_score(text, confidence, debug_info)
            candidate = {
                "attempt": attempt_index,
                "frame": frame,
                "text": text,
                "confidence": confidence,
                "debug_info": debug_info,
                "score": score,
                "failure_reason": debug_info.get("failure_reason", ""),
            }
            candidates.append(candidate)

            if best is None or candidate["score"] > best["score"]:
                best = candidate

            if text and confidence >= good_confidence and is_reasonable_speed(text):
                break

            if attempt_index < attempts and delay_seconds > 0:
                time.sleep(delay_seconds)

        value = text_to_float(best["text"])
        if not return_debug:
            return value

        info = self._recognition_info(
            best["frame"],
            best["text"],
            best["confidence"],
            best["debug_info"],
        )
        info["retry_attempts"] = len(candidates)
        info["retry_selected_attempt"] = best["attempt"]
        info["retry_candidates"] = " | ".join(
            (
                f"{item['attempt']}:{item['text'] if item['text'] else '--'}"
                f":{item['confidence']:.2f}:{item['failure_reason']}"
            )
            for item in candidates
        )
        return value, info

    def recognize(self, return_debug=False):
        frame = self.capture_frame()
        text, confidence, debug_info = recognize_wind_speed(
            frame,
            self.quad_points,
            return_debug=return_debug,
            manual_digit_boxes=self.manual_digit_boxes,
        )
        value = text_to_float(text)

        if not return_debug:
            return value

        return value, self._recognition_info(frame, text, confidence, debug_info)

    def recognize_stable(self, return_debug=False):
        frame = self.capture_frame()
        raw_text, confidence, debug_info = recognize_wind_speed(
            frame,
            self.quad_points,
            return_debug=return_debug,
            manual_digit_boxes=self.manual_digit_boxes,
        )
        stable_text = self.stabilizer.update(raw_text, confidence)
        value = text_to_float(stable_text)

        if not return_debug:
            return value

        return value, {
            "text": stable_text,
            "raw_text": raw_text,
            "raw_value": text_to_float(raw_text),
            "confidence": confidence,
            "quad_points": self.quad_points,
            "frame": frame,
            "preview": debug_info["preview"],
            "digit_summary": debug_info["digit_summary"],
            "digit_box_summary": debug_info["digit_box_summary"],
            "digit_boxes": debug_info["digit_boxes"],
            "digit_images": debug_info["digit_images"],
            "rectified": debug_info["rectified"],
            "rectified_gray": debug_info["rectified_gray"],
            "rectified_enhanced": debug_info["rectified_enhanced"],
            "rectified_binary": debug_info["rectified_binary"],
            "failure_reason": debug_info["failure_reason"],
        }

    def recognize_stable_with_retry(
        self,
        attempts=3,
        delay_seconds=0.10,
        return_debug=False,
        good_confidence=RETRY_GOOD_CONFIDENCE,
    ):
        raw_value, info = self.recognize_with_retry(
            attempts=attempts,
            delay_seconds=delay_seconds,
            return_debug=True,
            good_confidence=good_confidence,
        )
        raw_text = info["text"]
        confidence = info["confidence"]
        stable_text = self.stabilizer.update(raw_text, confidence)
        value = text_to_float(stable_text)

        if not return_debug:
            return value

        info = dict(info)
        info["text"] = stable_text
        info["raw_text"] = raw_text
        info["raw_value"] = raw_value
        return value, info

    def reset_stabilizer(self):
        self.stabilizer.reset()

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def recognize_wind_speed_once(
    camera_index=CAMERA_INDEX,
    roi_file=ROI_FILE,
    quad_points=None,
    warmup_frames=8,
    return_debug=False,
):
    """
    Capture one frame from the camera and recognize the current wind speed.

    Returns a float when recognition succeeds, otherwise None. When
    return_debug=True, returns (value, info), where info contains the raw text,
    confidence, selected ROI, captured frame, and debug preview image.
    """
    points = quad_points if quad_points is not None else load_quad(roi_file)
    manual_digit_boxes = load_digit_boxes(roi_file)
    if points is None:
        raise RuntimeError("No saved ROI. Run this file with --select-roi first.")

    frame = capture_frame(camera_index=camera_index, warmup_frames=warmup_frames)
    text, confidence, debug_info = recognize_wind_speed(
        frame,
        points,
        return_debug=return_debug,
        manual_digit_boxes=manual_digit_boxes,
    )
    value = text_to_float(text)

    if not return_debug:
        return value

    return value, {
        "text": text,
        "confidence": confidence,
        "quad_points": points,
        "frame": frame,
        "preview": debug_info["preview"],
        "digit_summary": debug_info["digit_summary"],
        "digit_box_summary": debug_info["digit_box_summary"],
        "digit_boxes": debug_info["digit_boxes"],
        "digit_images": debug_info["digit_images"],
        "rectified": debug_info["rectified"],
        "rectified_gray": debug_info["rectified_gray"],
        "rectified_enhanced": debug_info["rectified_enhanced"],
        "rectified_binary": debug_info["rectified_binary"],
        "failure_reason": debug_info["failure_reason"],
    }


def read_wind_speed(
    camera_index=CAMERA_INDEX,
    roi_file=ROI_FILE,
    quad_points=None,
    warmup_frames=30,
    retry_attempts=5,
    retry_delay_seconds=0.10,
    return_debug=False,
):
    """
    Public one-shot wind-speed read with fresh-frame retry enabled by default.

    Returns a float when recognition succeeds, otherwise None. When
    return_debug=True, returns (value, info), where info includes the selected
    frame, retry metadata, confidence, and OCR debug images.
    """
    with WindSpeedOCR(
        camera_index=camera_index,
        roi_file=roi_file,
        quad_points=quad_points,
        warmup_frames=warmup_frames,
    ) as reader:
        return reader.recognize_with_retry(
            attempts=retry_attempts,
            delay_seconds=retry_delay_seconds,
            return_debug=return_debug,
        )


def rect_to_ratio_box(rect, width, height):
    x1, y1, x2, y2 = rect
    x1, x2 = sorted((max(0, min(int(x1), width - 1)), max(0, min(int(x2), width - 1))))
    y1, y2 = sorted((max(0, min(int(y1), height - 1)), max(0, min(int(y2), height - 1))))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None

    return (
        x1 / float(width),
        y1 / float(height),
        (x2 + 1) / float(width),
        (y2 + 1) / float(height),
    )


def select_and_save_digit_boxes(frame, quad_points, roi_file=ROI_FILE):
    rectified = warp_quad(frame, quad_points)
    selection_state = {
        "boxes": load_digit_boxes(roi_file) or [],
        "dragging": False,
        "start": None,
        "current": None,
        "saved": False,
    }

    def mouse_callback(event, x, y, flags, param):
        height, width = rectified.shape[:2]
        point = clamp_point(x, y, width, height)

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(param["boxes"]) >= len(DIGIT_BOXES):
                return
            param["dragging"] = True
            param["start"] = point
            param["current"] = (*point, *point)
        elif event == cv2.EVENT_MOUSEMOVE and param["dragging"]:
            sx, sy = param["start"]
            param["current"] = (sx, sy, point[0], point[1])
        elif event == cv2.EVENT_LBUTTONUP and param["dragging"]:
            sx, sy = param["start"]
            ratio_box = rect_to_ratio_box((sx, sy, point[0], point[1]), width, height)
            if ratio_box is not None:
                param["boxes"].append(ratio_box)
                param["boxes"] = sorted(param["boxes"], key=lambda box: box[0])
            param["dragging"] = False
            param["start"] = None
            param["current"] = None

    cv2.namedWindow(PREVIEW_WINDOW)
    cv2.setMouseCallback(PREVIEW_WINDOW, mouse_callback, selection_state)
    print(
        f"Drag {len(DIGIT_BOXES)} digit boxes on the rectified ROI. "
        "Press s to save, u to undo, r to clear, q to quit."
    )

    try:
        while True:
            display_frame = rectified.copy()
            draw_digit_boxes_overlay(
                display_frame,
                selection_state["boxes"],
                selection_state["current"],
            )
            tip = (
                f"Drag digit boxes ({len(selection_state['boxes'])}/{len(DIGIT_BOXES)}), "
                "s save, u undo, r clear, q quit"
            )
            cv2.putText(
                display_frame,
                tip,
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )
            cv2.imshow(PREVIEW_WINDOW, display_frame)

            key = cv2.waitKey(20) & 0xFF
            if key == ord("s"):
                if len(selection_state["boxes"]) == len(DIGIT_BOXES):
                    save_digit_boxes(selection_state["boxes"], roi_file)
                    selection_state["saved"] = True
                    print(f"Saved digit boxes to {roi_file}")
                    break
                print(f"Need {len(DIGIT_BOXES)} digit boxes before saving.")
            elif key == ord("u"):
                if selection_state["boxes"]:
                    selection_state["boxes"].pop()
            elif key == ord("r"):
                selection_state["boxes"].clear()
                clear_saved_digit_boxes(roi_file)
                print("Cleared saved digit boxes.")
            elif key == ord("q"):
                break
    finally:
        cv2.destroyWindow(PREVIEW_WINDOW)

    return load_digit_boxes(roi_file)


def select_saved_digit_boxes(camera_index=CAMERA_INDEX, roi_file=ROI_FILE, warmup_frames=8):
    quad_points = load_quad(roi_file)
    if quad_points is None:
        raise RuntimeError("No saved ROI. Run this file with --select-roi first.")

    frame = capture_frame(camera_index=camera_index, warmup_frames=warmup_frames)
    return select_and_save_digit_boxes(frame, quad_points, roi_file)


def select_and_save_roi(camera_index=CAMERA_INDEX, roi_file=ROI_FILE, select_digits=True):
    selection_state = {
        "points": [],
        "frame": None,
        "saved_frame": None,
        "saved": False,
        "saved_points": load_quad(roi_file),
    }

    def mouse_callback(event, x, y, flags, param):
        frame = param["frame"]
        if frame is None or event != cv2.EVENT_LBUTTONDOWN:
            return

        frame_height, frame_width = frame.shape[:2]
        point = clamp_point(x, y, frame_width, frame_height)
        if len(selection_state["points"]) >= 4:
            return

        selection_state["points"].append(point)
        if len(selection_state["points"]) == 4:
            if is_valid_quad(selection_state["points"]):
                ordered = order_quad_points(selection_state["points"]).astype(int)
                saved_points = [tuple(point) for point in ordered]
                save_quad(saved_points, roi_file)
                selection_state["saved_points"] = saved_points
                selection_state["saved_frame"] = frame.copy()
                selection_state["saved"] = True
                print(f"Saved ROI to {roi_file}")
            else:
                print("Invalid ROI. Please select 4 corners again.")
            selection_state["points"].clear()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera_index}.")

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, selection_state)
    print("Click the 4 corners of the wind-speed display. Press q to quit, r to clear.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from camera.")

            selection_state["frame"] = frame
            display_frame = frame.copy()

            saved_points = selection_state["saved_points"]
            if saved_points is not None:
                draw_quad_overlay(display_frame, saved_points, closed=True, point_numbers=False)

            draw_quad_overlay(
                display_frame,
                selection_state["points"],
                closed=False,
                point_numbers=True,
            )
            tip = f"Click 4 points ({len(selection_state['points'])}/4), r clear, q quit"
            cv2.putText(
                display_frame,
                tip,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or selection_state["saved"]:
                break
            if key == ord("r"):
                selection_state["points"].clear()
                selection_state["saved_points"] = None
                clear_saved_quad(roi_file)
                print("Cleared saved ROI.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    saved_points = load_quad(roi_file)
    if select_digits and saved_points is not None and selection_state["saved_frame"] is not None:
        select_and_save_digit_boxes(selection_state["saved_frame"], saved_points, roi_file)

    return saved_points


def test_once(camera_index=CAMERA_INDEX, roi_file=ROI_FILE, show_preview=True, warmup_frames=20):
    with WindSpeedOCR(
        camera_index=camera_index,
        roi_file=roi_file,
        warmup_frames=warmup_frames,
    ) as reader:
        value, info = reader.recognize(return_debug=True)

    print(f"{ROI_LABEL}: {value if value is not None else '(empty)'}")
    print(f"raw={info['text'] if info['text'] else '--'} conf={info['confidence']:.2f}")

    if show_preview:
        cv2.imshow(PREVIEW_WINDOW, info["preview"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return value


def main():
    parser = argparse.ArgumentParser(description="Capture one frame and recognize 7-segment wind speed.")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="OpenCV camera index.")
    parser.add_argument("--roi-file", default=ROI_FILE, help="Path to saved ROI JSON.")
    parser.add_argument("--select-roi", action="store_true", help="Select and save the 4-point display ROI.")
    parser.add_argument("--select-digits", action="store_true", help="Select and save the 4 digit boxes inside the saved ROI.")
    parser.add_argument("--no-preview", action="store_true", help="Do not show the debug preview window.")
    parser.add_argument("--warmup-frames", type=int, default=20, help="Frames to discard before testing.")
    args = parser.parse_args()

    if args.select_roi:
        select_and_save_roi(
            camera_index=args.camera,
            roi_file=args.roi_file,
            select_digits=True,
        )
        return

    if args.select_digits:
        select_saved_digit_boxes(
            camera_index=args.camera,
            roi_file=args.roi_file,
            warmup_frames=args.warmup_frames,
        )
        return

    test_once(
        camera_index=args.camera,
        roi_file=args.roi_file,
        show_preview=not args.no_preview,
        warmup_frames=args.warmup_frames,
    )


if __name__ == "__main__":
    main()
