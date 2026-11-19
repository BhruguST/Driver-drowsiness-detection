"""Swin-T eye-state classifier for a trained MRL Eye checkpoint.

The checkpoint is the source of truth for architecture, input size,
normalization and class order. Metadata saved by the training notebook is
adopted automatically, so a checkpoint trained with ``OPEN=0, CLOSED=1`` (or
the reverse) is interpreted correctly without editing this file.

TRAINING/INFERENCE CONTRACT (see ``docs/swin_training_contract.md``):

  * architecture        : timm ``swin_tiny_patch4_window7_224``, num_classes=2
  * input               : 224 x 224 RGB eye crop, square region (see eye_crop.py)
  * preprocessing       : BGR -> RGB -> cv2.resize(224, 224, INTER_LINEAR)
                          -> float32 / 255 -> normalize(mean, std) -> CHW
  * normalization       : checkpoint ``image_mean`` / ``image_std`` or
                          ``normalization``; ImageNet values are the fallback.
  * class index mapping : checkpoint ``class_mapping`` (name -> index) or
                          ``class_names`` (index -> name). The local constants
                          below are only a fallback for bare ``state_dict`` files.
  * checkpoint format   : either a bare ``state_dict`` or a dict such as::

        {
            "model_state_dict": model.state_dict(),
            "model_name": "swin_tiny_patch4_window7_224",
            "image_size": 224,
            "class_mapping": {"OPEN": 0, "CLOSED": 1},
            "image_mean": (0.485, 0.456, 0.406),
            "image_std": (0.229, 0.224, 0.225),
        }

``torch`` and ``timm`` are optional dependencies (``uv sync --extra ml``) and
are imported lazily inside :meth:`SwinEyeClassifier.load`, so the rest of the
project keeps working without them. If ``models/swin_best.onnx`` exists and
``onnxruntime`` is installed, inference uses ONNX Runtime (roughly 3x faster on
CPU) and ``timm`` is not needed; the ``.pt`` file is still read for metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODEL_NAME = "swin_tiny_patch4_window7_224"
INPUT_SIZE = 224
NUM_CLASSES = 2

# NOTE: placeholders - must match the Colab training pipeline.
CLASS_NAMES: tuple[str, str] = ("closed", "open")
CLASS_INDEX_TO_LABEL: dict[int, str] = {0: "closed", 1: "open"}

# NOTE: placeholders - must match the Colab training pipeline.
NORMALIZE_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
NORMALIZE_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)

# NOTE: must match the resampling used during training.
RESIZE_INTERPOLATION = cv2.INTER_LINEAR

CHECKPOINT_PATH = Path(__file__).resolve().parent / "models" / "swin_best.pt"
DEFAULT_DEVICE = "cpu"

# ONNX Runtime is only used when the exported model file exists. Two threads
# benchmark faster than 4 on this laptop (P-cores vs E-core scheduling).
ONNX_NUM_THREADS = 2


@dataclass(frozen=True)
class Prediction:
    """Result of one eye-state classification."""

    label: str
    confidence: float
    scores: tuple[float, ...] = ()


def preprocess_eye_image(eye_crop_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR eye crop to a normalized CHW ``float32`` tensor.

    This function is intentionally torch-free so the preprocessing can be
    tested before any checkpoint or ML dependency exists. The exact same steps
    must be used in the Colab training/evaluation pipeline.
    """
    if not isinstance(eye_crop_bgr, np.ndarray) or eye_crop_bgr.size == 0:
        raise ValueError("eye crop must be a non-empty numpy array")

    if eye_crop_bgr.ndim == 2:
        crop_rgb = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_GRAY2RGB)
    elif eye_crop_bgr.ndim == 3 and eye_crop_bgr.shape[2] == 4:
        crop_rgb = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGRA2RGB)
    elif eye_crop_bgr.ndim == 3 and eye_crop_bgr.shape[2] == 3:
        crop_rgb = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"unsupported eye crop shape: {eye_crop_bgr.shape}")

    resized = cv2.resize(
        crop_rgb,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=RESIZE_INTERPOLATION,
    )
    scaled = resized.astype(np.float32) / 255.0
    normalized = (scaled - NORMALIZE_MEAN) / NORMALIZE_STD
    return np.ascontiguousarray(
        np.transpose(normalized, (2, 0, 1)).astype(np.float32)
    )


def validate_checkpoint_metadata(metadata: dict[str, Any]) -> list[str]:
    """Check checkpoint metadata against this module's configuration.

    Raises ``ValueError`` for incompatibilities that make the checkpoint
    unusable (wrong architecture or input size, invalid class names).
    Returns a list of warnings for values that would silently degrade
    predictions (different normalization or class ordering).
    """
    warnings: list[str] = []

    arch = metadata.get("arch", metadata.get("model_name"))
    if arch is not None and str(arch) != MODEL_NAME:
        raise ValueError(
            f"checkpoint architecture {arch!r} does not match {MODEL_NAME!r}"
        )

    input_size = metadata.get("input_size", metadata.get("image_size"))
    if input_size is not None and int(input_size) != INPUT_SIZE:
        raise ValueError(
            f"checkpoint input_size {input_size!r} does not match {INPUT_SIZE}"
        )

    class_names = metadata.get("class_names")
    if class_names is None and isinstance(metadata.get("class_mapping"), dict):
        try:
            class_names = [
                str(name)
                for name, _ in sorted(
                    metadata["class_mapping"].items(),
                    key=lambda item: int(item[1]),
                )
            ]
        except (TypeError, ValueError):
            class_names = None
    if class_names is not None:
        names = tuple(str(name).lower() for name in class_names)
        if len(names) != NUM_CLASSES or set(names) != {"open", "closed"}:
            raise ValueError(
                f"checkpoint class_names {class_names!r} must contain exactly "
                "'open' and 'closed'"
            )
        if names != CLASS_NAMES:
            warnings.append(
                f"checkpoint class order {names!r} differs from local default "
                f"{CLASS_NAMES!r}; the checkpoint order will be used"
            )

    normalization = metadata.get("normalization")
    if isinstance(normalization, dict):
        mean = normalization.get("mean")
        std = normalization.get("std")
    else:
        mean = metadata.get("image_mean")
        std = metadata.get("image_std")
    if mean is not None or std is not None:
        if mean is not None and not np.allclose(
            np.asarray(mean, dtype=np.float32), NORMALIZE_MEAN, atol=1e-6
        ):
            warnings.append(
                f"checkpoint normalization mean {mean!r} differs from local "
                f"{NORMALIZE_MEAN.tolist()!r}; update swin_classifier.py"
            )
        if std is not None and not np.allclose(
            np.asarray(std, dtype=np.float32), NORMALIZE_STD, atol=1e-6
        ):
            warnings.append(
                f"checkpoint normalization std {std!r} differs from local "
                f"{NORMALIZE_STD.tolist()!r}; update swin_classifier.py"
            )

    return warnings


def _split_checkpoint(checkpoint: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract ``(state_dict, metadata)`` from a saved checkpoint."""
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a state_dict dictionary")
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            metadata = {name: item for name, item in checkpoint.items() if name != key}
            return value, metadata
    return checkpoint, {}


class SwinEyeClassifier:
    """CPU-friendly wrapper around a fine-tuned Swin-T eye-state model.

    Loading is lazy and idempotent: constructing the object never imports
    torch or touches the checkpoint, so the pipeline can start in EAR mode
    even when neither is installed.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH
        self.device_name = device
        self._torch: Any = None
        self._model: Any = None
        self._device: Any = None
        self._session: Any = None
        self._onnx_input_name: str | None = None
        self._backend: str | None = None
        self._class_index_to_label = dict(CLASS_INDEX_TO_LABEL)
        self._load_attempted = False
        self._load_error: str | None = None

    @property
    def checkpoint_exists(self) -> bool:
        return self.checkpoint_path.is_file()

    @property
    def onnx_path(self) -> Path:
        return self.checkpoint_path.with_suffix(".onnx")

    @property
    def backend(self) -> str | None:
        return self._backend

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def available(self) -> bool:
        if self._backend is None and not self._load_attempted:
            self.load()
        return self._backend is not None

    @property
    def class_index_to_label(self) -> dict[int, str]:
        return dict(self._class_index_to_label)

    def load(self) -> bool:
        """Load the checkpoint. Returns True on success, never raises."""
        if self._backend is not None:
            return True

        self._load_attempted = True

        if not self.checkpoint_exists:
            self._load_error = (
                f"checkpoint not found at {self.checkpoint_path} "
                "(Swin-T weights have not been trained/connected yet)"
            )
            return False

        try:
            import torch
        except ImportError as exc:
            self._load_error = (
                f"torch is not installed ({exc}); run `uv sync --extra ml`"
            )
            return False

        try:
            checkpoint = self._torch_load(torch)
            state_dict, metadata = _split_checkpoint(checkpoint)

            for message in validate_checkpoint_metadata(metadata):
                print(f"[swin_classifier] WARNING: {message}")
            self._adopt_checkpoint_class_names(metadata)

            if self.onnx_path.is_file():
                if self._load_onnx():
                    self._torch = torch
                    self._load_error = None
                    return True

            import timm

            model = timm.create_model(
                MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES
            )
            cleaned: dict[str, Any] = {}
            for key, value in state_dict.items():
                if not torch.is_tensor(value):
                    continue
                if key.startswith("module."):
                    key = key[len("module.") :]
                cleaned[key] = value

            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            if missing:
                raise RuntimeError(
                    f"checkpoint is missing {len(missing)} required weight(s), "
                    f"e.g. {missing[:3]}"
                )
            if unexpected:
                print(
                    f"[swin_classifier] WARNING: ignored {len(unexpected)} "
                    "unexpected checkpoint key(s)"
                )

            model.eval()
            device = self._resolve_device(torch)
            model.to(device)
            if device.type == "cpu":
                model.to(memory_format=torch.channels_last)
            self._torch = torch
            self._model = model
            self._backend = "torch"
            self._load_error = None
            print(
                f"[swin_classifier] Loaded Swin-T checkpoint (torch, "
                f"channels_last) from {self.checkpoint_path}"
            )
            return True
        except Exception as exc:  # noqa: BLE001 - surface any load failure as text
            self._model = None
            self._session = None
            self._backend = None
            self._load_error = f"failed to load checkpoint: {exc}"
            return False

    def predict(self, eye_crop_bgr: np.ndarray) -> Prediction:
        """Classify one eye crop. Raises if no valid checkpoint is loaded."""
        if not self.available:
            raise RuntimeError(
                f"Swin-T classifier unavailable: {self._load_error or 'not loaded'}. "
                "No prediction was made."
            )

        tensor = preprocess_eye_image(eye_crop_bgr)
        if self._session is not None:
            logits = self._session.run(
                None, {self._onnx_input_name: tensor[np.newaxis, ...]}
            )[0]
            probabilities = self._softmax_numpy(logits)
        else:
            with self._torch.inference_mode():
                batch = self._torch.from_numpy(tensor).unsqueeze(0)
                if self._device is not None and self._device.type == "cpu":
                    batch = batch.contiguous(
                        memory_format=self._torch.channels_last
                    )
                logits = self._model(batch)
                probabilities = self._torch.softmax(logits, dim=1)[0].cpu().tolist()

        if len(probabilities) != NUM_CLASSES:
            raise RuntimeError(
                f"model returned {len(probabilities)} scores, expected {NUM_CLASSES}"
            )

        index = int(np.argmax(probabilities))
        label = self._class_index_to_label.get(index)
        if label is None:
            raise RuntimeError(f"no class label configured for index {index}")
        return Prediction(
            label=label,
            confidence=float(probabilities[index]),
            scores=tuple(float(score) for score in probabilities),
        )

    def _load_onnx(self) -> bool:
        """Load the exported ONNX model; return False so torch can be used."""
        onnx_path = self.onnx_path
        try:
            import onnxruntime as ort
        except ImportError:
            print(
                "[swin_classifier] onnxruntime is not installed; using the "
                f"slower torch backend for {onnx_path.name}"
            )
            return False

        try:
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = ONNX_NUM_THREADS
            session_options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                str(onnx_path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            self._session = session
            self._onnx_input_name = session.get_inputs()[0].name
            self._backend = "onnxruntime"
            print(
                f"[swin_classifier] Loaded ONNX model from {onnx_path} "
                f"({ONNX_NUM_THREADS} threads)"
            )
            return True
        except Exception as exc:  # noqa: BLE001 - fall back to torch
            print(
                f"[swin_classifier] WARNING: could not load {onnx_path.name} "
                f"({exc}); falling back to torch"
            )
            self._session = None
            self._onnx_input_name = None
            self._backend = None
            return False

    @staticmethod
    def _softmax_numpy(logits: Any) -> list[float]:
        scores = np.asarray(logits, dtype=np.float32).reshape(-1)
        scores = np.exp(scores - scores.max())
        return (scores / scores.sum()).tolist()

    def _torch_load(self, torch: Any) -> Any:
        try:
            return torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            # Older PyTorch versions do not support the weights_only argument.
            return torch.load(self.checkpoint_path, map_location="cpu")

    def _resolve_device(self, torch: Any) -> Any:
        device = torch.device(self.device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(
                "[swin_classifier] WARNING: CUDA requested but unavailable; using CPU"
            )
            device = torch.device("cpu")
        self._device = device
        return device

    def _adopt_checkpoint_class_names(self, metadata: dict[str, Any]) -> None:
        class_mapping = metadata.get("class_mapping")
        if isinstance(class_mapping, dict):
            index_to_label: dict[int, str] = {}
            for name, index in class_mapping.items():
                try:
                    index_to_label[int(index)] = str(name).lower()
                except (TypeError, ValueError):
                    continue
            if (
                len(index_to_label) == NUM_CLASSES
                and set(index_to_label.values()) == {"open", "closed"}
            ):
                self._class_index_to_label = index_to_label
                mapping = ", ".join(
                    f"{index}={name}"
                    for index, name in sorted(index_to_label.items())
                )
                print(
                    "[swin_classifier] Using class order from checkpoint "
                    f"class_mapping: {mapping}"
                )
                return

        class_names = metadata.get("class_names")
        if not class_names:
            return
        names = tuple(str(name).lower() for name in class_names)
        self._class_index_to_label = {
            index: name for index, name in enumerate(names)
        }
        mapping = ", ".join(
            f"{index}={name}" for index, name in self._class_index_to_label.items()
        )
        print(f"[swin_classifier] Using class order from checkpoint: {mapping}")


def _self_check() -> None:
    print(f"model name      : {MODEL_NAME}")
    print(f"input size      : {INPUT_SIZE}")
    print(f"classes         : {CLASS_NAMES}")
    print(f"checkpoint path : {CHECKPOINT_PATH}")
    print(f"checkpoint found: {CHECKPOINT_PATH.is_file()}")
    classifier = SwinEyeClassifier()
    print(f"onnx path       : {classifier.onnx_path}")
    print(f"onnx found      : {classifier.onnx_path.is_file()}")
    print(f"available       : {classifier.available}")
    print(f"backend         : {classifier.backend}")
    if classifier.load_error:
        print(f"reason          : {classifier.load_error}")


if __name__ == "__main__":
    _self_check()
