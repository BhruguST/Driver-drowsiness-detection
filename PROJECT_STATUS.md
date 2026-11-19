# Project context / handoff prompt (resume here)

This file is **not a status report anymore**. It is the context prompt for the
coding assistant. Read this first when the user returns with a retrained
Swin-T checkpoint, then follow the resume workflow below.

---

## 0. Role and rules

You are continuing a drowsy-driver detection mini-project. The user will bring
a **retrained Swin-T checkpoint** (`models/swin_best.pt`) and expects you to
integrate and verify it without re-discovery work.

Rules:

- Do **not** retrain unless the user explicitly asks.
- Do **not** hardcode class flips, thresholds, or predictions to make results
  look good. Fix root causes.
- Use checkpoint metadata as the source of truth for class order, input size,
  and normalization.
- The user must be present for webcam tests. Ask them to watch the **Camera**
  window; never claim a live test passed from logs alone.
- Do not commit, push, or rewrite git history unless explicitly asked.
- Keep the existing EAR, MAR, PERCLOS, fusion, and alarm behavior unless a
  measured bug requires a change.

---

## 1. Environment

- OS: Windows 10/11, PowerShell 5.1.
- Project root: `C:\Users\Bhrugu\projects\drowsy_project`.
- Python: 3.11.9 in `.venv`; run unbuffered:
  `.\.venv\Scripts\python.exe -u ...`
- CPU-only laptop (i5-1335U). Camera loop ~24 FPS at 640x480.
- App window is titled **Camera**; quit with `q`.
- MediaPipe model: `face_landmarker.task` in the project root.
- Alarm: `alarm.mp3` (loads with pygame 2.6.1 on this machine).
- Optional ML deps: `uv sync --extra ml` (`torch`, `timm`, `onnx`,
  `onnxruntime`, `onnxscript`).
- ONNX Runtime is configured for **2 threads** (`ONNX_NUM_THREADS = 2`); 4
  threads was erratic on this CPU.

---

## 2. Current state (last verified 2026-11-18)

- Checkpoint exists: `models/swin_best.pt` (~330 MB).
  - Metadata: `val_accuracy = 0.9937`, epoch 13/15.
  - `model_name = swin_tiny_patch4_window7_224`
  - `image_size = 224`
  - `class_mapping = {"OPEN": 0, "CLOSED": 1}`
  - `image_mean = (0.485, 0.456, 0.406)`
  - `image_std = (0.229, 0.224, 0.225)`
  - Split provenance is **not** recorded in the repo; do not claim a
    subject-wise split unless the user confirms it.
- ONNX export exists: `models/swin_best.onnx` + `models/swin_best.onnx.data`.
  - Verified against PyTorch: max abs logit difference `1.19e-6`.
  - ~0.24 s/eye on CPU with 2 threads vs ~0.73 s/eye for PyTorch.
- Test suite: `python test_swin_setup.py` -> **15/15 pass** (2026-11-18).
- Live SWIN closure test passed twice on 2026-11-18:
  - open-eye score `0.000`;
  - sustained closure -> `ALARM: sustained eye closure`;
  - reopen -> `ALARM cleared: awake`;
  - closed-eye sample mean `0.84`, 89% of frames above 0.5 after the transition.
- Slow-blink/dozing tracker is implemented and **unit-tested**, but the live
  slow-blink alarm has not been confirmed by the user yet.
- EAR remains the default mode; SWIN is selectable with `--mode SWIN`.

---

## 3. Non-negotiable implementation invariants

1. **Class order comes from the checkpoint.**
   - Prefer `class_mapping` (name -> index); accept `class_names` in index order.
   - The old default `{0: "closed", 1: "open"}` was inverted relative to the
     current checkpoint and caused "always drowsy". Never assume label order.
   - If a new checkpoint has no mapping, print a clear warning and use the
     local fallback; do not silently flip.
2. **Normalization/architecture come from the checkpoint.**
   - Read `model_name`/`arch`, `image_size`, `image_mean`/`image_std`.
   - Current defaults are `swin_tiny_patch4_window7_224`, 224x224, RGB,
     ImageNet mean/std, `/255`.
3. **Eye crops must stay square and undistorted.**
   - Use the six EAR eye landmarks, expand by `DEFAULT_PADDING = 0.10`, make the
     region square, shift it inside the frame instead of squashing, then resize
     to 224x224.
   - Wide crops resized to 224x224 were the second root cause of poor live
     accuracy.
4. **Preprocessing:** OpenCV gives BGR; convert to RGB; resize with
   `cv2.INTER_LINEAR`; CHW; float32; /255; ImageNet normalization.
5. **ONNX first, PyTorch fallback.**
   - If `models/swin_best.onnx` exists, use ONNX Runtime; both `.onnx` and
     `.onnx.data` must be present.
   - If ONNX fails, warn and use channels-last PyTorch (`torch.inference_mode`).
6. **SWIN runs on a background worker.**
   - `SwinInferenceWorker` replaces pending work (newest frame wins), exposes a
     sequence number per completed prediction, and resets on face loss.
   - Camera loop must never block on inference.
7. **Never mix EAR and SWIN in one session.**
   - While SWIN has no result (warm-up), treat eye state as unknown; do not
     substitute EAR. Mixing distributions caused false startup alarms.
8. **PERCLOS is time-weighted.**
   - Add one sample per **new measurement**, not per camera frame.
   - 60 s window, 30 s minimum warm-up.
   - Each sample is valid until the next sample.
9. **Slow-blink/dozing tracker.**
   - Slow closure: `0.4 <= duration < 1.5` s.
   - Require `SLOW_BLINK_MIN_SAMPLES = 2` consecutive fresh closed measurements
     (filters 2 Hz SWIN quantization).
   - Alarm after `SLOW_BLINK_ALARM_COUNT = 3` slow blinks within
     `SLOW_BLINK_WINDOW = 60` s.
   - Clear `SLOW_BLINK_CLEAR_AFTER = 20` s after the last slow blink.
   - Microsleep (>=1.5 s) is handled by the sustained-closure alarm, not counted
     as a slow blink.
10. **Fusion/alarm ownership.**
    - The alarm belongs to the signal that triggered it and stops when that
      signal clears.
    - If a yawn or slow-blink alarm escalates to a sustained closure, switch the
      trigger to `"eye"` and keep the alarm on.
    - Face loss resets timers, PERCLOS history, the closure tracker, the SWIN
      generation, and the alarm.

---

## 4. Files and responsibilities

| File | Responsibility |
|---|---|
| `drowsy_pipeline.py` | Main app: MediaPipe -> EAR/SWIN -> closure, slow-blink, MAR, PERCLOS -> fusion -> pygame alarm. HUD shows mode, EAR, MAR, PERCLOS, SWIN score, slow-blink count, status, FPS. Flags: `--mode {EAR,SWIN}`, `--checkpoint`, `--swin-threshold`. |
| `eye_crop.py` | Square eye crops from EAR landmarks, 10% padding, centred/shifted inside frame, rejects bad crops. |
| `swin_classifier.py` | Metadata adoption, preprocessing, ONNX Runtime backend, channels-last PyTorch fallback, `predict()` -> `(label, confidence)`. |
| `export_swin_onnx.py` | Export checkpoint to ONNX, verify logits vs PyTorch, benchmark CPU latency, print class mapping/normalization. |
| `test_swin_setup.py` | 15 checks: imports, preprocessing, square crops, metadata adoption, graceful failures, real inference, worker sequencing, slow-blink tracker. |
| `webcam_test.py` | Camera + 478 landmarks + FPS test rig. |
| `README.md` | Setup/build guide and current SWIN integration docs. |
| `docs/swin_training_contract.md` | Training/inference contract: architecture, crop geometry, metadata keys, datasets, ONNX export. |
| `models/swin_best.pt` | Trained checkpoint (ignored by git). |
| `models/swin_best.onnx` + `.onnx.data` | ONNX export + external weights (ignored by git; both required). |
| `alarm.mp3` | Alarm audio. |
| `face_landmarker.task` | MediaPipe Face Landmarker model. |

---

## 5. Resume workflow when the user returns with a retrained checkpoint

1. **Read the new metadata before touching code.**
   ```powershell
   .\.venv\Scripts\python.exe -u -c "
   import torch
   ckpt = torch.load(r'models/swin_best.pt', map_location='cpu', weights_only=False)
   print(type(ckpt))
   print(ckpt.keys() if isinstance(ckpt, dict) else 'bare state_dict')
   for key in ('model_name','arch','image_size','class_mapping','class_names','image_mean','image_std','val_accuracy','dataset'):
       if isinstance(ckpt, dict) and key in ckpt:
           print(key, '=', ckpt[key])
   "
   ```
   - If class order/labels changed, the loader should adopt them generically.
   - If the new model is **3-class** (`open`/`partial`/`closed`) or regresses
     eyelid aperture, stop and discuss the semantics before wiring it in. Do not
     map 3 classes onto 2 with a hardcoded guess.
2. **Back up the working checkpoint/ONNX** outside `models/` before replacing
   anything.
3. **Place the new `models/swin_best.pt`** and delete the old `.onnx` +
   `.onnx.data` so a stale export cannot be loaded.
4. **Run diagnostics:**
   ```powershell
   .\.venv\Scripts\python.exe -u test_swin_setup.py
   .\.venv\Scripts\python.exe -u swin_classifier.py
   ```
   - Confirm the printed class order/normalization match the new metadata.
   - Confirm backend is `onnxruntime` only after re-export; before that it should
     use the PyTorch fallback.
5. **Export and verify ONNX:**
   ```powershell
   .\.venv\Scripts\python.exe -u export_swin_onnx.py
   ```
   - Require a small max logit difference (the previous export was `1.19e-6`).
   - Ensure both `.onnx` and `.onnx.data` exist.
   - If latency is worse than ~0.5 s/eye, investigate before the live demo.
6. **Run the test suite again**, then launch the app:
   ```powershell
   .\.venv\Scripts\python.exe -u test_swin_setup.py
   .\.venv\Scripts\python.exe -u drowsy_pipeline.py --mode SWIN
   ```
7. **Live verification protocol (user must watch the Camera window):**
   - eyes open ~5 s -> no alarm, SWIN score near 0;
   - close and hold ~3 s -> `ALARM: sustained eye closure`;
   - reopen -> `ALARM cleared: awake`;
   - 3 slow blinks (~0.8 s closed, ~2 s open) -> `ALARM: repeated slow blinks`;
   - wait ~20 s -> slow-blink alarm clears;
   - optional: yawn, look away/face lost, and `q` to quit.
   - If the user is on another tab or not watching, the test is invalid; rerun.
8. **Only after live data exists, calibrate thresholds.** Prefer measured
   distributions over guesses:
   - SWIN cutoff: ROC/Youden's J on labeled open/closed frames.
   - Slow-blink constants: labeled simulated dozing vs normal blinking.
   - PERCLOS: event-level false alarms/min and detection latency.
   - Keep changes in the constants at the top of `drowsy_pipeline.py`; do not
     hide tuning in per-frame hacks.
9. **Report honestly:** distinguish checkpoint metadata accuracy from measured
   live accuracy; report the domain gap and any missed/false alarms.
10. **Update docs only for real behavior/command changes.** Do not claim live
    verification that did not happen.

---

## 6. Retraining guidance (user plans to retrain)

- Static eye-state datasets (MRL Eye, CEW) teach open/closed appearance, not
  blink dynamics or dozing.
- Temporal drowsiness datasets:
  - **UTA-RLDD** — RGB videos, ~60 subjects, alert/low vigilance/drowsy.
  - **NTHU-DDD** — driving videos, RGB + IR, sleepy/yawning/nodding annotations.
  - **DROZY** — IR video + physiological signals, sleep-deprivation labels.
  - **YawDD** — yawning only.
- Prefer a **subject-wise split**, never random.
- Save metadata with every checkpoint:
  ```python
  {
      "model_state_dict": model.state_dict(),
      "model_name": "swin_tiny_patch4_window7_224",
      "image_size": 224,
      "class_mapping": {"OPEN": 0, "CLOSED": 1},
      "image_mean": (0.485, 0.456, 0.406),
      "image_std": (0.229, 0.224, 0.225),
      "dataset": "...",
      "val_accuracy": ...,
  }
  ```
- If training for partial closures/dozing, consider 3 classes
  (`open`/`partial`/`closed`) or eyelid-aperture regression, and update the
  pipeline generically (no hardcoded class indices).
- MRL Eye is grayscale/near-infrared; the webcam is RGB. Always validate on
  webcam clips in addition to the test split.

---

## 7. Known pitfalls (learned the hard way)

- Inverted class order was the main "always drowsy" bug. Never assume labels.
- Wide eye crops squashed to 224x224 were the second bug. Keep crops square.
- Do not substitute EAR while SWIN warms up; it caused false startup alarms.
- A 10 s PERCLOS warm-up made a single 3 s closure look like 30-46% PERCLOS and
  kept the alarm on. Use the 30 s warm-up.
- Counting every camera frame as a PERCLOS/SWIN sample inflates closed time.
  Count only fresh measurements.
- At ~2 SWIN updates/s, a single closed sample cannot distinguish a normal blink
  from a slow one; require two consecutive fresh closed samples.
- `swin_best.onnx` needs its sibling `.onnx.data`; copying only the `.onnx`
  breaks loading.
- ONNX Runtime with 4 threads was slower/erratic than 2 on this CPU.
- The app can take ~10 s to open the Camera window while the 112 MB ONNX model
  loads; do not assume it crashed immediately.
- `Start-Process` with redirected output sometimes errors in this shell; a
  `.bat` launcher invoked through WMI works. Recreate one if needed.
- Tests load the real checkpoint and take a few seconds; do not run two apps
  using the camera at the same time.

---

## 8. Quick command reference

```powershell
cd C:\Users\Bhrugu\projects\drowsy_project
.venv\Scripts\activate

# Diagnostics
python test_swin_setup.py
python swin_classifier.py

# Run
python drowsy_pipeline.py
python drowsy_pipeline.py --mode SWIN
python drowsy_pipeline.py --mode SWIN --swin-threshold 0.55

# Re-export after retraining
python export_swin_onnx.py

# Camera/landmark sanity check
python webcam_test.py
```

---

## 9. What "done" means when the user returns

- New checkpoint loads with its own class order and normalization.
- ONNX export matches PyTorch and uses both model files.
- `test_swin_setup.py` passes all checks.
- Live SWIN closure alarm and slow-blink/dozing behavior are verified with the
  user at the camera.
- Any reported accuracy is measured or clearly labeled as checkpoint metadata.
- The user knows what remains (evaluation, threshold tuning, report/demo).
