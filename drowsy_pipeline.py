"""Webcam drowsiness detection pipeline.

Signals:
  * Signal 1 - eye state: EAR by default, optionally the Swin-T classifier.
  * Signal 2 - yawning: mouth aspect ratio (MAR).
  * Signal 3 - PERCLOS: rolling percentage of closed-eye frames.

Eye detection modes (``--mode`` / ``EYE_DETECTION_MODE``):
  * ``EAR``  - always available, no model required (default).
  * ``SWIN`` - requires trained weights at ``models/swin_best.pt`` and the
               optional ML dependencies (``uv sync --extra ml``). If the
               checkpoint is absent or invalid, the pipeline prints why and
               falls back to EAR instead of crashing.
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pygame
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from eye_crop import DEFAULT_PADDING, extract_eye_crops
from swin_classifier import CHECKPOINT_PATH, SwinEyeClassifier

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308]  # top lip, bottom lip, left corner, right corner

# ---- Configuration ---------------------------------------------------------

EYE_DETECTION_MODE = "EAR"  # "EAR" or "SWIN"
VALID_EYE_MODES = ("EAR", "SWIN")

EAR_THRESHOLD = 0.20
CLOSED_DURATION_LIMIT = 1.5  # seconds - sustained closure = drowsy
MAR_THRESHOLD = 0.6  # mouth aspect ratio above this = mouth open wide
YAWN_DURATION_LIMIT = 1.0  # seconds - mouth must stay open this long to count as a yawn
PERCLOS_WINDOW = 60  # seconds of history to keep
PERCLOS_THRESHOLD = 0.15  # 15% of the window spent closed = drowsy
# Ignore PERCLOS until half the rolling window exists. With a shorter span a
# single 2-3 s closure looks like 30%+ "closed", which fires PERCLOS and keeps
# the alarm on after the eyes reopen.
PERCLOS_MIN_WINDOW = 30  # seconds

# Slow-blink / dozing pattern. Normal blinks are shorter than SLOW_BLINK_MIN;
# closures between SLOW_BLINK_MIN and CLOSED_DURATION_LIMIT count as slow
# blinks, and repeated slow blinks are an early dozing sign. Microsleep-length
# closures are handled by the sustained-closure alarm instead.
SLOW_BLINK_MIN = 0.4  # seconds
# Consecutive fresh closed measurements required. At ~2 Hz SWIN updates this
# stops a single sample from turning a normal blink into a "slow" one.
SLOW_BLINK_MIN_SAMPLES = 2
SLOW_BLINK_ALARM_COUNT = 3
SLOW_BLINK_WINDOW = 60.0  # seconds
# Clear the repeated-slow-blink alarm this long after the last slow blink.
SLOW_BLINK_CLEAR_AFTER = 20.0  # seconds

# Swin mode only: probability cutoff for "eye closed". Calibrated on live
# samples; still worth re-tuning during evaluation.
SWIN_CLOSED_THRESHOLD = 0.5
# Fraction of extra padding around the eye landmark bounding box.
EYE_CROP_PADDING = DEFAULT_PADDING

MODEL_ASSET_PATH = "face_landmarker.task"
ALARM_PATH = "alarm.mp3"


# ---- Signal helpers (EAR/MAR math unchanged) -------------------------------


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        points.append(np.array([lm.x * w, lm.y * h]))
    p1, p2, p3, p4, p5, p6 = points
    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return (vertical1 + vertical2) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks, mouth_indices, w, h):
    top, bottom, left, right = [
        np.array([landmarks[i].x * w, landmarks[i].y * h])
        for i in mouth_indices
    ]
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    return vertical / horizontal


def closed_probability_from_crops(
    classifier: SwinEyeClassifier,
    left_crop: np.ndarray | None,
    right_crop: np.ndarray | None,
) -> float | None:
    """Return the highest closed-eye probability across the two eye crops.

    ``None`` means no usable crop was classified, so the caller can fall back
    to EAR instead of guessing.
    """
    closed_scores: list[float] = []
    for crop in (left_crop, right_crop):
        if crop is None:
            continue
        try:
            prediction = classifier.predict(crop)
        except Exception as exc:  # noqa: BLE001 - a single bad frame must not stop the app
            print(f"[pipeline] Swin prediction failed: {exc}")
            continue
        closed_scores.append(
            prediction.confidence
            if prediction.label == "closed"
            else 1.0 - prediction.confidence
        )

    return max(closed_scores) if closed_scores else None


def predict_eye_closed(
    classifier: SwinEyeClassifier,
    frame: np.ndarray,
    landmarks: Any,
    threshold: float = SWIN_CLOSED_THRESHOLD,
) -> tuple[bool | None, float | None]:
    """Return ``(is_closed, closed_probability)`` for one frame.

    Kept as a synchronous helper for tests/scripts; the live pipeline uses
    :class:`SwinInferenceWorker` so slow CPU inference does not stall capture.
    """
    left_crop, right_crop = extract_eye_crops(
        frame, landmarks, padding=EYE_CROP_PADDING
    )
    score = closed_probability_from_crops(classifier, left_crop, right_crop)
    if score is None:
        return None, None
    return score >= threshold, score


class SwinInferenceWorker:
    """Classify eye crops on a background thread; the newest result wins.

    Swin-T on a laptop CPU takes a few hundred milliseconds per frame. Running
    that inline would drop the camera loop to a crawl and distort the timers,
    so the capture loop submits crops and keeps using the latest completed
    prediction. Pending work is replaced rather than queued, so latency never
    grows if inference is slower than capture.
    """

    def __init__(self, classifier: SwinEyeClassifier) -> None:
        self._classifier = classifier
        self._condition = threading.Condition()
        self._pending: tuple[
            np.ndarray | None, np.ndarray | None, int
        ] | None = None
        self._result: tuple[int, float | None, int] | None = None
        self._sequence = 0
        self._generation = 0
        self._stop = False
        self._thread = threading.Thread(
            target=self._run, name="swin-inference", daemon=True
        )
        self._thread.start()

    def reset(self) -> int:
        """Invalidate pending and in-flight results (called when face is lost)."""
        with self._condition:
            self._generation += 1
            self._pending = None
            self._result = None
            return self._generation

    def submit(
        self,
        left_crop: np.ndarray | None,
        right_crop: np.ndarray | None,
        generation: int,
    ) -> None:
        with self._condition:
            self._pending = (left_crop, right_crop, generation)
            self._condition.notify()

    def latest(self, generation: int) -> tuple[float | None, int | None]:
        """Return ``(closed_score, sequence)`` for the newest measurement.

        ``sequence`` changes only when a new prediction completes, which lets
        the caller add exactly one PERCLOS sample per real measurement instead
        of counting the same held score on every camera frame.
        """
        with self._condition:
            if self._result is not None and self._result[0] == generation:
                return self._result[1], self._result[2]
            return None, None

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._pending = None
            self._condition.notify()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._stop and self._pending is None:
                    self._condition.wait()
                if self._stop:
                    return
                left_crop, right_crop, generation = self._pending
                self._pending = None

            score = closed_probability_from_crops(
                self._classifier, left_crop, right_crop
            )

            with self._condition:
                if not self._stop:
                    self._sequence += 1
                    self._result = (generation, score, self._sequence)


class ClosureTracker:
    """Track eye closures and flag repeated slow blinks.

    A single long closure is handled by the sustained-closure alarm. Dozing
    often shows up earlier as several closures that are too long for a normal
    blink but shorter than a microsleep. This tracker records those, requiring
    ``min_samples`` fresh closed observations so a normal blink cannot become
    "slow" just because a 2 Hz SWIN score was held between frames.
    """

    def __init__(
        self,
        slow_min: float = SLOW_BLINK_MIN,
        slow_max: float = CLOSED_DURATION_LIMIT,
        min_samples: int = SLOW_BLINK_MIN_SAMPLES,
        alarm_count: int = SLOW_BLINK_ALARM_COUNT,
        window: float = SLOW_BLINK_WINDOW,
        clear_after: float = SLOW_BLINK_CLEAR_AFTER,
    ) -> None:
        self.slow_min = slow_min
        self.slow_max = slow_max
        self.min_samples = min_samples
        self.alarm_count = alarm_count
        self.window = window
        self.clear_after = clear_after
        self.alert = False
        self._closed_since: float | None = None
        self._closed_samples = 0
        self._events: deque[float] = deque()

    @property
    def slow_blink_count(self) -> int:
        return len(self._events)

    def closed_duration(self, now: float) -> float | None:
        if self._closed_since is None:
            return None
        return max(0.0, now - self._closed_since)

    def reset(self) -> None:
        self._closed_since = None
        self._closed_samples = 0
        self._events.clear()
        self.alert = False

    def update(
        self, now: float, closed: bool | None, measured: bool
    ) -> bool:
        """Feed one observation and return whether the pattern alarm is active.

        ``closed=None`` means the eye state is unknown (for example while the
        SWIN model is warming up); any in-progress closure is discarded.
        """
        if closed is None:
            self._closed_since = None
            self._closed_samples = 0
        elif closed:
            if self._closed_since is None:
                self._closed_since = now
                self._closed_samples = 1
            elif measured:
                self._closed_samples += 1
        elif self._closed_since is not None:
            duration = now - self._closed_since
            if (
                self.slow_min <= duration < self.slow_max
                and self._closed_samples >= self.min_samples
            ):
                self._events.append(now)
            self._closed_since = None
            self._closed_samples = 0

        while self._events and now - self._events[0] > self.window:
            self._events.popleft()

        self.alert = (
            len(self._events) >= self.alarm_count
            and now - self._events[-1] <= self.clear_after
        )
        return self.alert


def resolve_eye_mode(
    requested_mode: str, classifier: SwinEyeClassifier
) -> str:
    """Return the eye-detection mode actually used, with a clear log message."""
    requested_mode = requested_mode.upper()

    if requested_mode == "SWIN":
        if classifier.available:
            print("[pipeline] Eye detection mode: SWIN (Swin-T checkpoint loaded)")
            return "SWIN"
        print("[pipeline] Swin-T weights have not been trained or connected yet.")
        print(f"[pipeline] Checkpoint: {classifier.checkpoint_path}")
        print(f"[pipeline] Reason: {classifier.load_error}")
        print("[pipeline] Falling back to EAR eye detection.")
        return "EAR"

    print("[pipeline] Eye detection mode: EAR")
    return "EAR"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Webcam-based drowsy driver detection"
    )
    parser.add_argument(
        "--mode",
        default=EYE_DETECTION_MODE,
        type=str.upper,
        choices=VALID_EYE_MODES,
        help=(
            "Eye-state detector: EAR (default) or SWIN "
            "(requires models/swin_best.pt)"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=Path,
        help=f"Path to the Swin-T checkpoint (default: {CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--swin-threshold",
        default=SWIN_CLOSED_THRESHOLD,
        type=float,
        help=(
            "Closed-eye probability cutoff for SWIN mode "
            f"(default: {SWIN_CLOSED_THRESHOLD})"
        ),
    )
    return parser.parse_args(argv)


# ---- Main loop -------------------------------------------------------------


def run(
    mode: str,
    checkpoint_path: Path | None,
    swin_threshold: float = SWIN_CLOSED_THRESHOLD,
) -> None:
    classifier = SwinEyeClassifier(checkpoint_path or CHECKPOINT_PATH)
    eye_mode = resolve_eye_mode(mode, classifier)
    if eye_mode == "SWIN":
        print(f"[pipeline] SWIN closed-eye threshold: {swin_threshold:.2f}")

    swin_worker = SwinInferenceWorker(classifier) if eye_mode == "SWIN" else None
    swin_generation = swin_worker.reset() if swin_worker else 0

    pygame.mixer.init()
    alarm_sound = pygame.mixer.Sound(ALARM_PATH)

    cap = cv2.VideoCapture(0)
    prev_time = time.perf_counter()

    base_options = python.BaseOptions(model_asset_path=MODEL_ASSET_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.VIDEO
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    closure_tracker = ClosureTracker()
    mouth_open_since = None
    eye_closure_alert = False
    slow_blink_alert = False
    yawn_alert = False
    perclos_alert = False
    prev_eye_alert = False
    prev_slow_blink_alert = False
    prev_yawn_alert = False
    prev_perclos_alert = False
    alarm_on = False
    alarm_trigger: str | None = None
    last_swin_sequence: int | None = None

    # each entry: (timestamp, was_eye_closed)
    closure_history: deque[tuple[float, bool]] = deque()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read from camera")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                h, w, _ = frame.shape
                now_t = time.perf_counter()

                # ---- Signal 1: eye closure ----
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                swin_closed, swin_confidence = None, None
                swin_sequence = None
                if swin_worker is not None:
                    left_crop, right_crop = extract_eye_crops(
                        frame, landmarks, padding=EYE_CROP_PADDING
                    )
                    swin_worker.submit(left_crop, right_crop, swin_generation)
                    swin_confidence, swin_sequence = swin_worker.latest(
                        swin_generation
                    )
                    if swin_confidence is not None:
                        swin_closed = swin_confidence >= swin_threshold

                if swin_worker is not None:
                    # Never mix classifiers in one session: EAR and SWIN have
                    # different score distributions, and an EAR disagreement
                    # during model warm-up can fire a false closure alarm.
                    # Treat the state as unknown until the first prediction.
                    eye_closed_now = swin_closed
                    eye_state_source = "SWIN"
                else:
                    eye_closed_now = avg_ear < EAR_THRESHOLD
                    eye_state_source = "EAR"

                if eye_closed_now is None:
                    new_eye_measurement = False
                elif swin_worker is None:
                    new_eye_measurement = True
                else:
                    new_eye_measurement = (
                        swin_closed is not None
                        and swin_sequence != last_swin_sequence
                    )
                if new_eye_measurement and swin_sequence is not None:
                    last_swin_sequence = swin_sequence

                slow_blink_alert = closure_tracker.update(
                    now_t, eye_closed_now, new_eye_measurement
                )
                closed_for = closure_tracker.closed_duration(now_t)
                eye_closure_alert = (
                    closed_for is not None
                    and closed_for >= CLOSED_DURATION_LIMIT
                )

                # ---- Signal 2: yawning ----
                mar = mouth_aspect_ratio(landmarks, MOUTH, w, h)
                mouth_open_now = mar > MAR_THRESHOLD

                if mouth_open_now:
                    if mouth_open_since is None:
                        mouth_open_since = now_t
                    yawn_alert = (
                        now_t - mouth_open_since
                    ) >= YAWN_DURATION_LIMIT
                else:
                    mouth_open_since = None
                    yawn_alert = False

                # ---- Signal 3: PERCLOS over a rolling window ----
                # In SWIN mode predictions arrive a few times per second; add
                # one sample per new prediction (not per camera frame) so a
                # held score cannot inflate the closed fraction. In EAR mode
                # every camera frame is a real measurement.
                if eye_closed_now is not None and new_eye_measurement:
                    closure_history.append((now_t, eye_closed_now))

                while (
                    closure_history
                    and now_t - closure_history[0][0] > PERCLOS_WINDOW
                ):
                    closure_history.popleft()

                # Time-weighted fraction of the window spent closed. Each
                # sample is treated as valid until the next sample, which is
                # exactly how held SWIN scores behave.
                closed_time = 0.0
                total_time = 0.0
                for (t0, was_closed), (t1, _) in pairwise(closure_history):
                    delta = t1 - t0
                    total_time += delta
                    if was_closed:
                        closed_time += delta
                perclos = closed_time / total_time if total_time > 0 else 0.0
                history_span = (
                    closure_history[-1][0] - closure_history[0][0]
                    if len(closure_history) >= 2
                    else 0.0
                )
                perclos_alert = (
                    history_span >= PERCLOS_MIN_WINDOW
                    and perclos >= PERCLOS_THRESHOLD
                )

                # ---- Fusion ----
                # The alarm belongs to the signal that started it and stops the
                # moment that signal clears (e.g. as soon as a yawn ends). A
                # signal that was already active cannot keep the alarm on or
                # restart it; it must clear and cross its threshold again.
                eye_rising = eye_closure_alert and not prev_eye_alert
                slow_blink_rising = slow_blink_alert and not prev_slow_blink_alert
                yawn_rising = yawn_alert and not prev_yawn_alert
                perclos_rising = perclos_alert and not prev_perclos_alert
                prev_eye_alert = eye_closure_alert
                prev_slow_blink_alert = slow_blink_alert
                prev_yawn_alert = yawn_alert
                prev_perclos_alert = perclos_alert

                if not alarm_on:
                    if eye_rising:
                        alarm_on, alarm_trigger = True, "eye"
                    elif slow_blink_rising:
                        alarm_on, alarm_trigger = True, "slow_blink"
                    elif yawn_rising:
                        alarm_on, alarm_trigger = True, "yawn"
                    elif perclos_rising:
                        alarm_on, alarm_trigger = True, "perclos"

                    if alarm_on:
                        alarm_sound.play(loops=-1)
                        if alarm_trigger == "eye":
                            print("ALARM: sustained eye closure")
                        elif alarm_trigger == "slow_blink":
                            print(
                                "ALARM: repeated slow blinks "
                                f"({closure_tracker.slow_blink_count} in "
                                f"{SLOW_BLINK_WINDOW:.0f}s)"
                            )
                        elif alarm_trigger == "yawn":
                            print("ALARM: yawn detected")
                        else:
                            print(f"ALARM: PERCLOS high ({perclos:.2f})")
                else:
                    trigger_cleared = (
                        (alarm_trigger == "eye" and not eye_closure_alert)
                        or (
                            alarm_trigger == "slow_blink"
                            and not slow_blink_alert
                        )
                        or (alarm_trigger == "yawn" and not yawn_alert)
                        or (alarm_trigger == "perclos" and not perclos_alert)
                    )
                    if trigger_cleared:
                        if alarm_trigger == "slow_blink" and eye_closure_alert:
                            # The closure escalated to a microsleep; keep the
                            # alarm on the stronger signal.
                            alarm_trigger = "eye"
                            print("ALARM: sustained eye closure")
                        elif alarm_trigger == "yawn" and eye_closure_alert:
                            # Eyes are still closed past the limit, so keep the
                            # alarm running on the stronger closure signal.
                            alarm_trigger = "eye"
                            print("ALARM: sustained eye closure")
                        elif alarm_trigger == "yawn" and slow_blink_alert:
                            alarm_trigger = "slow_blink"
                            print(
                                "ALARM: repeated slow blinks "
                                f"({closure_tracker.slow_blink_count} in "
                                f"{SLOW_BLINK_WINDOW:.0f}s)"
                            )
                        else:
                            ended_trigger = alarm_trigger
                            alarm_on, alarm_trigger = False, None
                            alarm_sound.stop()
                            if ended_trigger == "eye":
                                print("ALARM cleared: awake")
                            elif ended_trigger == "slow_blink":
                                print("ALARM cleared: blink rate normal")
                            elif ended_trigger == "yawn":
                                print("ALARM cleared: yawn finished")
                            else:
                                print("ALARM cleared: PERCLOS normal")

                status = "DROWSY" if alarm_on else "AWAKE"
                status_color = (0, 0, 255) if alarm_on else (0, 255, 0)

                cv2.putText(
                    frame,
                    f"Mode: {eye_state_source}  EAR: {avg_ear:.3f}  "
                    f"MAR: {mar:.2f}  PERCLOS: {perclos:.2f}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
                if swin_confidence is not None:
                    cv2.putText(
                        frame,
                        f"SWIN closed score: {swin_confidence:.2f}",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )
                cv2.putText(
                    frame,
                    f"Slow blinks: {closure_tracker.slow_blink_count}/"
                    f"{SLOW_BLINK_ALARM_COUNT} in {SLOW_BLINK_WINDOW:.0f}s",
                    (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255) if slow_blink_alert else (255, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    status,
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    status_color,
                    2,
                )

                for idx in LEFT_EYE + RIGHT_EYE + MOUTH:
                    lm = landmarks[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            else:
                # Face lost: reset per-face state so stale timers cannot fire
                # a false alarm when the face returns.
                if swin_worker is not None:
                    swin_generation = swin_worker.reset()
                    last_swin_sequence = None
                closure_history.clear()
                closure_tracker.reset()
                mouth_open_since = None
                eye_closure_alert = False
                slow_blink_alert = False
                yawn_alert = False
                perclos_alert = False
                prev_eye_alert = False
                prev_slow_blink_alert = False
                prev_yawn_alert = False
                prev_perclos_alert = False
                alarm_trigger = None
                if alarm_on:
                    alarm_on = False
                    alarm_sound.stop()
                    print("ALARM cleared: face not detected")

            now = time.perf_counter()
            delta = now - prev_time
            fps = 1 / delta if delta > 0 else 0.0
            prev_time = now

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if swin_worker is not None:
            swin_worker.close()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        pygame.mixer.quit()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(args.mode, args.checkpoint, args.swin_threshold)


if __name__ == "__main__":
    main()
