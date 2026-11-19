"""Eye-region cropping from MediaPipe Face Landmarker output.

The crop region is derived from the SAME six landmarks per eye that the EAR
calculation in ``drowsy_pipeline.py`` uses. The EAR math itself is not
modified: this module only turns those points into an eye image for the
Swin-T classifier.

The crop is square by default. Resizing a wide landmark box to the classifier's
224x224 input squashes the eye vertically, which does not match the square
training images and badly hurts live accuracy. A square region centred on the
eye keeps the eye's aspect ratio intact.

Output is a BGR ``numpy`` crop (same channel order as the OpenCV frame).
``None`` is returned for missing, non-finite, zero-area, or too-small regions
so malformed images are never passed to a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Left/right naming follows the existing EAR constants in drowsy_pipeline.py.
LEFT_EYE_LANDMARKS: tuple[int, ...] = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_LANDMARKS: tuple[int, ...] = (33, 160, 158, 133, 153, 144)

# Fraction of the eye bounding-box size added on each side.
DEFAULT_PADDING = 0.10
MIN_CROP_SIZE = 8


def _pixel_points(
    landmarks: Sequence[Any] | None,
    indices: Sequence[int],
    width: int,
    height: int,
) -> np.ndarray | None:
    """Convert landmark indices to pixel coordinates, or return None."""
    if landmarks is None:
        return None
    try:
        points = np.array(
            [
                [float(landmarks[index].x) * width, float(landmarks[index].y) * height]
                for index in indices
            ],
            dtype=np.float32,
        )
    except (IndexError, KeyError, TypeError, AttributeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[0] < 3 or not np.isfinite(points).all():
        return None
    return points


def extract_eye_crop(
    frame: np.ndarray | None,
    landmarks: Sequence[Any] | None,
    eye_indices: Sequence[int],
    padding: float = DEFAULT_PADDING,
    min_size: int = MIN_CROP_SIZE,
    square: bool = True,
) -> np.ndarray | None:
    """Return a BGR eye crop for one eye, or None if the region is unusable.

    With ``square=True`` (default) the crop is a square centred on the eye,
    kept inside the frame without changing its shape. ``square=False`` keeps
    the raw landmark-box aspect ratio and is only useful for experimentation.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None

    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return None

    points = _pixel_points(landmarks, eye_indices, width, height)
    if points is None:
        return None

    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    box_width = x_max - x_min
    box_height = y_max - y_min
    if box_width <= 0 or box_height <= 0:
        return None

    if square:
        side = min(
            max(box_width, box_height) * (1.0 + 2.0 * padding),
            float(width),
            float(height),
        )
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        x0 = min(max(0.0, center_x - side / 2.0), width - side)
        y0 = min(max(0.0, center_y - side / 2.0), height - side)
        x1 = x0 + side
        y1 = y0 + side
    else:
        pad_x = box_width * padding
        pad_y = box_height * padding
        x0 = max(0.0, x_min - pad_x)
        y0 = max(0.0, y_min - pad_y)
        x1 = min(float(width), x_max + pad_x)
        y1 = min(float(height), y_max + pad_y)

    x0, y0 = int(np.floor(x0)), int(np.floor(y0))
    x1, y1 = int(np.ceil(x1)), int(np.ceil(y1))

    if x1 - x0 < min_size or y1 - y0 < min_size:
        return None

    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return np.ascontiguousarray(crop)


def extract_eye_crops(
    frame: np.ndarray | None,
    landmarks: Sequence[Any] | None,
    padding: float = DEFAULT_PADDING,
    min_size: int = MIN_CROP_SIZE,
    square: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return ``(left_crop, right_crop)``; either side may be None."""
    left = extract_eye_crop(
        frame, landmarks, LEFT_EYE_LANDMARKS, padding, min_size, square
    )
    right = extract_eye_crop(
        frame, landmarks, RIGHT_EYE_LANDMARKS, padding, min_size, square
    )
    return left, right


def _self_check() -> None:
    """Print a quick summary of the module configuration."""
    print(f"module              : {Path(__file__).name}")
    print(f"left eye landmarks  : {LEFT_EYE_LANDMARKS}")
    print(f"right eye landmarks : {RIGHT_EYE_LANDMARKS}")
    print(f"padding             : {DEFAULT_PADDING}")
    print(f"min crop size       : {MIN_CROP_SIZE}")
    print("crop shape          : square, centred on the eye")


if __name__ == "__main__":
    _self_check()
