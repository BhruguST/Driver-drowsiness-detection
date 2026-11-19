"""Export the trained Swin-T eye-state checkpoint to ONNX.

Usage:

    python export_swin_onnx.py

Writes ``models/swin_best.onnx`` next to the checkpoint. The pipeline detects
that file automatically and uses ONNX Runtime, which is roughly 3x faster than
the torch CPU path on this laptop. The ``.pt`` file is still required because
it stores the class order and normalization metadata.

Export needs ``onnx`` and ``onnxscript``; inference needs ``onnxruntime``.
Install them with:

    uv pip install onnx onnxscript onnxruntime
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import timm

from swin_classifier import (
    CHECKPOINT_PATH,
    INPUT_SIZE,
    MODEL_NAME,
    NUM_CLASSES,
    ONNX_NUM_THREADS,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help=f"Trained checkpoint (default: {CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="ONNX output path (default: checkpoint with .onnx suffix)",
    )
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args(argv)


def load_state_dict(checkpoint: dict) -> tuple[dict, dict]:
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            metadata = {k: v for k, v in checkpoint.items() if k != key}
            return value, metadata
    return checkpoint, {}


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"checkpoint not found: {args.checkpoint}")
        return 1

    output_path = args.output or args.checkpoint.with_suffix(".onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        print("checkpoint must contain a state_dict dictionary")
        return 1
    state_dict, metadata = load_state_dict(checkpoint)

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    cleaned = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
        if torch.is_tensor(value)
    }
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"checkpoint is missing {len(missing)} weight(s), e.g. {missing[:3]}")
        return 1
    if unexpected:
        print(f"ignoring {len(unexpected)} unexpected checkpoint key(s)")
    model.eval()

    mapping = metadata.get("class_mapping")
    if mapping:
        print(f"checkpoint class mapping: {mapping}")
    image_mean = metadata.get("image_mean") or metadata.get(
        "normalization", {}
    ).get("mean")
    image_std = metadata.get("image_std") or metadata.get(
        "normalization", {}
    ).get("std")
    if image_mean is not None or image_std is not None:
        print(f"checkpoint normalization: mean={image_mean} std={image_std}")

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    started = time.perf_counter()
    try:
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            input_names=["input"],
            output_names=["logits"],
            opset_version=args.opset,
            do_constant_folding=True,
        )
    except ImportError as exc:
        print(
            f"ONNX export needs extra packages ({exc}). "
            "Run: uv pip install onnx onnxscript"
        )
        return 1
    print(f"exported in {time.perf_counter() - started:.1f}s -> {output_path}")

    with torch.inference_mode():
        expected = model(dummy).numpy()

    try:
        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = ONNX_NUM_THREADS
        session_options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(output_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
    except ImportError:
        print("onnxruntime is not installed; run `uv pip install onnxruntime`")
        return 1

    input_name = session.get_inputs()[0].name
    actual = session.run(None, {input_name: dummy.numpy()})[0]
    diff = float(np.abs(expected - actual).max())
    print(f"torch vs onnx max abs logit difference: {diff:.6g}")
    if diff > 1e-3:
        print("WARNING: ONNX output differs from torch; not using fast path")

    for _ in range(2):
        session.run(None, {input_name: dummy.numpy()})
    started = time.perf_counter()
    runs = 5
    for _ in range(runs):
        session.run(None, {input_name: dummy.numpy()})
    per_eye = (time.perf_counter() - started) / runs
    print(
        f"onnxruntime speed: {per_eye * 1000:.0f} ms/eye "
        f"(~{1 / (2 * per_eye):.1f} FPS for both eyes)"
    )
    print(f"model size: {output_path.stat().st_size / (1024 * 1024):.1f} MB")
    print("Now run: python test_swin_setup.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
