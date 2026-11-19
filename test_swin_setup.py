"""Verify the Swin-T integration scaffold without trained weights.

Run from the project root:

    python test_swin_setup.py

Checks imports, preprocessing, eye cropping, graceful behavior when the
checkpoint is missing, and real inference only if a checkpoint and the
optional ML dependencies are present. Exits non-zero on failure.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import drowsy_pipeline
import eye_crop
import swin_classifier


RESULTS: list[tuple[str, bool, str]] = []


def check(name: str) -> None:
    """Decorator that runs one check function and records its result."""

    def register(function):
        try:
            function()
        except Exception as exc:  # noqa: BLE001 - report every failure, keep running
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        else:
            RESULTS.append((name, True, ""))
        return function

    return register


def make_landmarks() -> list[SimpleNamespace]:
    """Build 478 neutral landmarks with realistic eye positions."""
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    left = [
        (0.30, 0.40),
        (0.32, 0.39),
        (0.34, 0.39),
        (0.38, 0.40),
        (0.34, 0.41),
        (0.32, 0.41),
    ]
    right = [
        (0.60, 0.40),
        (0.62, 0.39),
        (0.64, 0.39),
        (0.68, 0.40),
        (0.64, 0.41),
        (0.62, 0.41),
    ]
    for indices, points in (
        (eye_crop.LEFT_EYE_LANDMARKS, left),
        (eye_crop.RIGHT_EYE_LANDMARKS, right),
    ):
        for index, (x, y) in zip(indices, points):
            landmarks[index] = SimpleNamespace(x=x, y=y)
    return landmarks


@check("project imports with EAR as the default mode")
def test_imports() -> None:
    assert drowsy_pipeline.EYE_DETECTION_MODE == "EAR"
    assert drowsy_pipeline.VALID_EYE_MODES == ("EAR", "SWIN")
    assert callable(drowsy_pipeline.run)


@check("preprocessing produces a 3x224x224 float32 tensor")
def test_preprocess_shape() -> None:
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, size=(48, 72, 3), dtype=np.uint8)
    tensor = swin_classifier.preprocess_eye_image(crop)
    assert tensor.shape == (3, swin_classifier.INPUT_SIZE, swin_classifier.INPUT_SIZE)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


@check("preprocessing converts BGR to RGB and applies normalization")
def test_preprocess_values() -> None:
    white_bgr = np.full((40, 40, 3), 255, dtype=np.uint8)
    tensor = swin_classifier.preprocess_eye_image(white_bgr)
    expected = (np.ones(3, dtype=np.float32) - swin_classifier.NORMALIZE_MEAN) / (
        swin_classifier.NORMALIZE_STD
    )
    np.testing.assert_allclose(tensor[:, 0, 0], expected, rtol=1e-5)


@check("preprocessing rejects empty or invalid crops")
def test_preprocess_rejects_invalid() -> None:
    for bad in (None, np.zeros((0, 0, 3), dtype=np.uint8), "not an image"):
        try:
            swin_classifier.preprocess_eye_image(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {type(bad).__name__}")


@check("eye crops are extracted for both eyes")
def test_eye_crops_valid() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    left, right = eye_crop.extract_eye_crops(frame, make_landmarks())
    assert left is not None and right is not None
    assert left.ndim == 3 and left.shape[2] == 3
    assert min(left.shape[:2]) >= eye_crop.MIN_CROP_SIZE
    assert min(right.shape[:2]) >= eye_crop.MIN_CROP_SIZE


@check("eye crops are square so the classifier sees an undistorted eye")
def test_eye_crops_square() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    left, right = eye_crop.extract_eye_crops(frame, make_landmarks())
    assert left is not None and right is not None
    assert abs(left.shape[0] - left.shape[1]) <= 2
    assert abs(right.shape[0] - right.shape[1]) <= 2


@check("checkpoint metadata supplies class order and normalization")
def test_checkpoint_metadata() -> None:
    metadata = {
        "model_name": swin_classifier.MODEL_NAME,
        "image_size": swin_classifier.INPUT_SIZE,
        "class_mapping": {"OPEN": 0, "CLOSED": 1},
        "image_mean": (0.485, 0.456, 0.406),
        "image_std": (0.229, 0.224, 0.225),
    }
    warnings = swin_classifier.validate_checkpoint_metadata(metadata)
    assert any("class order" in warning for warning in warnings)
    classifier = swin_classifier.SwinEyeClassifier()
    classifier._adopt_checkpoint_class_names(metadata)
    assert classifier.class_index_to_label == {0: "open", 1: "closed"}


@check("eye crops reject missing, zero-area, and invalid landmarks")
def test_eye_crops_invalid() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert eye_crop.extract_eye_crops(frame, None) == (None, None)
    assert (
        eye_crop.extract_eye_crop(frame, [], eye_crop.LEFT_EYE_LANDMARKS) is None
    )
    assert (
        eye_crop.extract_eye_crop(None, make_landmarks(), eye_crop.LEFT_EYE_LANDMARKS)
        is None
    )
    flat = [SimpleNamespace(x=0.3, y=0.4) for _ in range(478)]
    assert (
        eye_crop.extract_eye_crop(frame, flat, eye_crop.LEFT_EYE_LANDMARKS) is None
    )


@check("preprocessing runs on a real extracted eye crop")
def test_preprocess_extracted_crop() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    left, _ = eye_crop.extract_eye_crops(frame, make_landmarks())
    assert left is not None
    tensor = swin_classifier.preprocess_eye_image(left)
    assert tensor.shape == (3, 224, 224)


@check("missing checkpoint is handled gracefully")
def test_missing_checkpoint() -> None:
    classifier = swin_classifier.SwinEyeClassifier()
    if classifier.checkpoint_exists:
        print("    (skipped: a checkpoint is already present)")
        return
    assert classifier.available is False
    assert classifier.load() is False
    assert classifier.load_error
    try:
        classifier.predict(np.zeros((32, 32, 3), dtype=np.uint8))
    except RuntimeError:
        pass
    else:
        raise AssertionError("predict() must refuse to run without a checkpoint")


@check("SWIN mode falls back to EAR when no checkpoint exists")
def test_swin_fallback() -> None:
    classifier = swin_classifier.SwinEyeClassifier()
    if classifier.checkpoint_exists:
        print("    (skipped: a checkpoint is already present)")
        return
    assert drowsy_pipeline.resolve_eye_mode("SWIN", classifier) == "EAR"
    assert drowsy_pipeline.resolve_eye_mode("EAR", classifier) == "EAR"


@check("corrupt checkpoint fails gracefully without crashing")
def test_corrupt_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "corrupt.pt"
        path.write_bytes(b"this is not a checkpoint")
        classifier = swin_classifier.SwinEyeClassifier(path)
        assert classifier.available is False
        assert classifier.load_error
        try:
            classifier.predict(np.zeros((32, 32, 3), dtype=np.uint8))
        except RuntimeError:
            pass
        else:
            raise AssertionError("predict() must refuse a corrupt checkpoint")


@check("checkpoint loads and predicts when weights are available")
def test_checkpoint_inference() -> None:
    classifier = swin_classifier.SwinEyeClassifier()
    if not classifier.checkpoint_exists:
        print("    (skipped: models/swin_best.pt not present yet)")
        return
    assert classifier.available, f"checkpoint exists but failed: {classifier.load_error}"
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    prediction = classifier.predict(crop)
    assert prediction.label in ("open", "closed")
    assert 0.0 <= prediction.confidence <= 1.0
    print(
        f"    backend: {classifier.backend}  "
        f"prediction: {prediction.label} "
        f"(confidence={prediction.confidence:.3f}, scores={prediction.scores})"
    )


@check("SWIN worker exposes one sequence number per completed prediction")
def test_swin_worker_sequence() -> None:
    classifier = swin_classifier.SwinEyeClassifier()
    if not classifier.checkpoint_exists:
        print("    (skipped: models/swin_best.pt not present yet)")
        return
    assert classifier.available, f"checkpoint exists but failed: {classifier.load_error}"

    worker = drowsy_pipeline.SwinInferenceWorker(classifier)
    try:
        assert worker.latest(0) == (None, None)
        generation = worker.reset()
        crop = np.zeros((64, 64, 3), dtype=np.uint8)

        worker.submit(crop, crop, generation)
        deadline = time.time() + 15.0
        score, sequence = None, None
        while time.time() < deadline:
            score, sequence = worker.latest(generation)
            if sequence is not None:
                break
            time.sleep(0.05)
        assert score is not None, "worker produced no prediction"
        assert sequence is not None
        assert 0.0 <= score <= 1.0
        assert worker.latest(generation) == (score, sequence)

        worker.submit(crop, crop, generation)
        deadline = time.time() + 15.0
        new_sequence = sequence
        while time.time() < deadline:
            _, new_sequence = worker.latest(generation)
            if new_sequence is not None and new_sequence > sequence:
                break
            time.sleep(0.05)
        assert new_sequence > sequence, "sequence must advance per prediction"
    finally:
        worker.close()


@check("closure tracker flags repeated slow blinks and ignores normal blinks")
def test_closure_tracker() -> None:
    tracker = drowsy_pipeline.ClosureTracker(
        slow_min=0.4,
        slow_max=1.5,
        min_samples=2,
        alarm_count=3,
        window=60.0,
        clear_after=20.0,
    )

    tracker.update(0.0, True, True)
    tracker.update(0.2, False, True)
    assert tracker.slow_blink_count == 0
    assert tracker.alert is False

    # One fresh SWIN sample could represent a longer hold; do not count it.
    tracker.update(1.0, True, True)
    tracker.update(1.5, False, True)
    assert tracker.slow_blink_count == 0

    tracker.update(2.0, True, True)
    tracker.update(2.5, True, True)
    tracker.update(3.0, False, True)
    assert tracker.slow_blink_count == 1
    assert tracker.alert is False

    tracker.update(4.0, True, True)
    tracker.update(4.8, True, True)
    tracker.update(5.2, False, True)
    tracker.update(6.0, True, True)
    tracker.update(6.8, True, True)
    tracker.update(7.2, False, True)
    assert tracker.slow_blink_count == 3
    assert tracker.alert is True

    # A microsleep-length closure belongs to the sustained-closure alarm.
    tracker.update(10.0, True, True)
    tracker.update(12.0, True, True)
    tracker.update(12.5, False, True)
    assert tracker.slow_blink_count == 3

    tracker.update(30.0, False, True)
    assert tracker.alert is False

    tracker.update(31.0, None, False)
    assert tracker.closed_duration(31.0) is None
    tracker.reset()
    assert tracker.slow_blink_count == 0
    assert tracker.alert is False


def main() -> int:
    failures = 0
    print("Swin-T integration checks")
    print("-" * 48)
    for name, ok, message in RESULTS:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}")
        if message:
            print(f"       {message}")
        if not ok:
            failures += 1
    print("-" * 48)
    print(f"{len(RESULTS) - failures} passed, {failures} failed")

    if not swin_classifier.CHECKPOINT_PATH.is_file():
        print(
            "\nNOTE: Swin-T has not been trained/connected yet. "
            f"Expected checkpoint: {swin_classifier.CHECKPOINT_PATH}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
