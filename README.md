# Drowsy Driver Detection — Mini Project

Webcam-based driver drowsiness detection using MediaPipe face landmarks, an Eye Aspect Ratio (EAR) baseline, and a fine-tuned Swin Transformer (Swin-T) eye-state classifier.

**Team:** BMSIT&M, 3-credit Mini Project (BCS506)

**Handoff context:** see [PROJECT_STATUS.md](PROJECT_STATUS.md) — it is now a context prompt for resuming work (including retrained-checkpoint integration), not a status report.

---

## 1. Project overview

**Updated scope:** drowsiness is judged from three signals together, not eye closure alone. A single sustained eye closure is a *late* sign of fatigue; yawning is an *early* sign, and PERCLOS captures a *gradual* decline — combining them matches the project's own claim of catching early signs of fatigue.

| Item | Detail |
|---|---|
| Core pipeline | Webcam → MediaPipe face landmarks → **three signals** (below) → fusion rule → audible alarm |
| Signal 1 — Eye closure | Eye crop → classify open/closed (EAR baseline or Swin-T) → sustained-closure timer (~1.5–2 s) plus repeated slow-closure (0.4–1.5 s) detection |
| Signal 2 — Yawning | Mouth landmarks → mouth aspect ratio → minimum-duration rule (so talking/smiling doesn't trigger it) |
| Signal 3 — PERCLOS | Percentage of time eyes were closed over a rolling window (e.g. last 60 s), not just one event |
| Fusion rule | Alarm triggers if ANY signal crosses its threshold, and stops as soon as that signal clears — exact thresholds tuned during Week 4 evaluation |
| Baseline | Eye Aspect Ratio (EAR), no machine learning |
| Main model | Swin-T (`swin_tiny_patch4_window7_224`), fine-tuned on MRL Eye — used for Signal 1 only |
| Out of scope | Night/infrared driving, testing in a moving car |

---

## 2. What runs where

Training needs a GPU; your laptop doesn't have one. Everything involving the live camera needs your laptop; Colab can't see a webcam. So the work splits cleanly:

| Task | Where | Why |
|---|---|---|
| Webcam capture, FPS check | **Local** | Colab has no access to your camera |
| MediaPipe face/eye landmarks | **Local** | Lightweight, needs the live camera |
| EAR calculation, closure timer, alarm | **Local** | Needs camera + speakers |
| Download/explore MRL Eye dataset | **Colab** | Keeps large files off your laptop, next to the GPU |
| Fine-tune Swin-T | **Colab** | Needs a GPU to train in minutes instead of days |
| Evaluate model (accuracy, F1, confusion matrix) | **Colab** | Fast there, data already loaded |
| Run trained model live on webcam | **Local** | Inference only — CPU is enough |
| Batch-test on recorded clips | Either | Simpler locally once everything is installed |
| Report, slides, code | **Local** | — |

---

## PART A — Local setup (your laptop)

### A1. Requirements

- Windows 10/11
- [Python 3.11](https://www.python.org/downloads/) (MediaPipe does not yet fully support 3.13)
- [uv](https://docs.astral.sh/uv/) — fast Python package/environment manager
- [VS Code](https://code.visualstudio.com/)
- [Git](https://git-scm.com/) (for sharing code with teammates)
- A working webcam and speakers

### A2. Clone or create the project folder

If you're starting fresh:

```powershell
cd C:\Users\<you>\projects
mkdir drowsy_project
cd drowsy_project
```

If you're a teammate joining an existing repo:

```powershell
git clone <repo-url>
cd drowsy_project
```

### A3. Create the environment and install dependencies

This project uses `uv` in project mode (`pyproject.toml` + `uv.lock`), not a plain `requirements.txt`.

```powershell
uv venv --python 3.11
.venv\Scripts\activate
uv sync
```

- `uv venv --python 3.11` creates `.venv` using Python 3.11.
- `uv sync` reads `pyproject.toml` / `uv.lock` and installs the exact same versions everyone on the team uses.

If you are setting up `pyproject.toml` for the first time instead of joining an existing repo:

```powershell
uv init
uv add opencv-python mediapipe numpy pygame
```

Verify the install:

```powershell
python -c "import cv2, mediapipe, numpy, pygame; print(cv2.__version__, mediapipe.__version__, numpy.__version__, pygame.__version__)"
```

### A4. Download the MediaPipe face landmark model

MediaPipe's current ("Tasks") API needs the model file downloaded separately. Save this into the project folder (next to your scripts, not in `.venv`):

```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

Check it landed correctly:

```powershell
dir face_landmarker.task
```

### A5. Local build order

Work through these in order. Each step should fully work before moving to the next.

> **File note:** Steps 1–2 live in `webcam_test.py`, your detection test rig — it proves the camera and MediaPipe work, and already detects all 478 face landmarks (eyes, mouth, everything). Steps 3 onward build the real pipeline in a new file (e.g. `drowsy_pipeline.py`), reading different subsets of those same landmarks rather than detecting anything new. You don't need a second detector.

1. **`webcam_test.py`** — open the camera, show live video, display an FPS counter, quit on `q`.
   **Functions used in this step:**

   | Function | What it does |
   |---|---|
   | `cv2.VideoCapture(0)` | Opens camera `0` (the laptop's built-in webcam) and returns a handle used for every later camera operation. |
   | `cap.read()` | Grabs one frame. Returns `(ok, frame)` — `ok` is `True`/`False` depending on whether the read succeeded, `frame` is the image as a grid of pixel numbers. |
   | `time.perf_counter()` | A high-precision clock, used to time how long each frame takes so FPS can be calculated. |
   | `cv2.putText(frame, text, pos, font, size, color, thickness)` | Draws text onto a frame — used here to show the live FPS counter. OpenCV colors are in **(Blue, Green, Red)** order, not RGB. |
   | `cv2.imshow(window_name, frame)` | Opens/updates a window showing the given frame. |
   | `cv2.waitKey(1)` | Waits up to 1 ms for a key press and lets the window redraw; without it the window freezes. `& 0xFF == ord("q")` checks whether the key pressed was `q`. |
   | `cap.release()` | Frees the camera so other programs can use it again. |
   | `cv2.destroyAllWindows()` | Closes any OpenCV windows still open. |

2. **Face landmarks** — load `face_landmarker.task`, detect the face each frame, draw all 478 landmark points.

   **Functions used in this step:**

   | Function | What it does |
   |---|---|
   | `python.BaseOptions(model_asset_path=...)` | Points MediaPipe at the downloaded `.task` model file on disk. |
   | `vision.FaceLandmarkerOptions(base_options=..., running_mode=...)` | Configures the detector. `running_mode=vision.RunningMode.VIDEO` tells it to expect a continuous stream of timestamped frames (so it can track the face between frames) rather than isolated images. |
   | `vision.FaceLandmarker.create_from_options(options)` | Builds the actual detector object, called **once**, before the camera loop — rebuilding it every frame would be slow and pointless. |
   | `mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)` | Wraps an OpenCV frame into the image format MediaPipe expects. |
   | `landmarker.detect_for_video(mp_image, timestamp_ms)` | Runs detection on one frame. Needs a millisecond timestamp (`int(time.time() * 1000)`) because VIDEO mode uses timing to track the face smoothly between calls. |
   | `result.face_landmarks` | A list with one entry per detected face (empty if none found). Each entry is a list of 478 points with `.x`, `.y`, `.z` — `.x`/`.y` are **fractions between 0 and 1** of the frame's width/height, not pixel positions. |
   | `cv2.circle(frame, (x, y), radius, color, thickness)` | Draws a small dot at a pixel position; `thickness=-1` fills the circle. Used to mark each landmark point after converting its fraction to a pixel coordinate (`x = int(point.x * w)`, `y = int(point.y * h)`). |
3. **EAR baseline** — isolate the eye landmark points, compute the Eye Aspect Ratio, print live values for open/closed eyes.

   **Concept/function used in this step:**

   | Item | What it does |
   |---|---|
   | EAR formula: `(\|p2−p6\| + \|p3−p5\|) / (2·\|p1−p4\|)` | Six points around one eye give a ratio of "vertical eye opening" to "eye width." Open eyes give a higher ratio; closing eyes drop the ratio toward zero. No machine learning involved — just distances between points. |
   | `numpy.linalg.norm(point_a - point_b)` | Computes the straight-line (Euclidean) distance between two landmark points — used for each `\|p_i − p_j\|` term in the formula above. |
   | Specific landmark indices (e.g. left eye `362, 385, 387, 263, 373, 380`) | MediaPipe's 478 points are numbered consistently; these specific index numbers correspond to the six points needed for the EAR formula on each eye. |

4. **Closure timer** — add a threshold and a ~1.5–2 second sustained-closure rule so normal blinks don't trigger anything. *(This is Signal 1, complete.)*

4b. **Yawning (Signal 2, core — not a stretch goal anymore)** — isolate mouth landmarks (e.g. inner lips `13, 14`, corners `78, 308`), compute a mouth aspect ratio the same way as EAR, and add a minimum-duration rule so talking/smiling isn't mistaken for a yawn.

4c. **PERCLOS (Signal 3, new)** — keep a rolling buffer of the last ~60 seconds of EAR/eye-state values, and compute the percentage of that window spent closed. This needs the eye-state history over time, not just the current frame's value, so it's a different kind of logic than the Step 4 sustained-closure timer even though both use the same EAR numbers.

4d. **Fusion rule** — combine Signals 1–3: trigger the alarm if ANY of (a) sustained closure, (b) a detected yawn, or (c) PERCLOS above its threshold is true. Log which signal fired, so your evaluation (Week 4) can report each signal's individual contribution.

   **Concept used in this step:**

   | Item | What it does |
   |---|---|
   | EAR threshold (e.g. `~0.2`) | A cutoff below which the eye is considered "closed" for that frame. Tuned from your own observed EAR values (Step 3). |
   | A running timer (e.g. storing `time.perf_counter()` when the eye first drops below threshold) | Measures how long the eye has stayed below the threshold continuously. A normal blink (~0.1–0.4 s) ends before the ~1.5–2 s cutoff and never triggers the alarm; sustained closure crosses it. |

5. **Alarm** — play a sound (via `pygame.mixer`) when the fusion rule (4d) fires on any signal; stop it as soon as that signal clears (eyes reopen, the yawn ends, or PERCLOS returns to normal).

   **Functions used in this step:**

   | Function | What it does |
   |---|---|
   | `pygame.mixer.init()` | Starts pygame's audio system — call once, before the loop. |
   | `pygame.mixer.Sound("alarm.wav")` | Loads a sound file from disk into memory, ready to play. |
   | `sound.play(loops=-1)` | Plays the sound. `loops=-1` repeats it continuously until stopped, so the alarm doesn't just beep once. |
   | `sound.stop()` | Stops playback — called once the eyes reopen. |

6. **Integration with Swin-T** (after Part B) — this only replaces **Signal 1's** classifier: swap the EAR check for the trained model's prediction, keep the same timer/fusion/alarm logic, add a switch between EAR and Swin modes. Signals 2 (yawning) and 3 (PERCLOS) are unaffected by this step — they don't use Swin at all.

   **Functions used in this step:**

   | Function | What it does |
   |---|---|
   | `timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=2)` | Rebuilds the same Swin-T architecture used in training, so the saved weights have something to load into. |
   | `model.load_state_dict(torch.load("swin_best.pt", map_location="cpu"))` | Loads your fine-tuned weights from the checkpoint file into the model. `map_location="cpu"` is needed because the laptop has no GPU, even though the checkpoint was trained on one. |
   | `model.eval()` | Switches the model to inference mode (turns off training-only behaviours like dropout) — always call this before using a model to make predictions rather than train it. |
   | `torch.no_grad():` (used as `with torch.no_grad():`) | Tells PyTorch not to track gradients during inference, since gradients are only needed for training — this makes prediction faster and uses less memory. |
   | `model(image_tensor)` | Runs the cropped, preprocessed eye image through the model and returns raw scores for "open" vs "closed," which are then compared to decide the prediction. |

### A6. Running scripts

Always activate the environment first:

```powershell
cd C:\Users\<you>\projects\drowsy_project
.venv\Scripts\activate
python webcam_test.py
```

### A7. Sharing code with teammates (GitHub)

```powershell
git init                      # only if not already a repo
git add .
git commit -m "Describe what changed"
git push
```

Make sure `.gitignore` includes `.venv/` — never commit the virtual environment, datasets, model checkpoints, or test videos. Share large files via the team's Google Drive folder instead.

Teammates set up their own copy with:

```powershell
git clone <repo-url>
cd drowsy_project
uv sync
```

---

## PART B — Google Colab (training)

Colab provides a free GPU, which your laptop doesn't have, needed to fine-tune Swin-T in minutes instead of days.

### B1. Open a notebook

1. Go to [colab.research.google.com](https://colab.research.google.com) and sign in.
2. **New notebook**, rename it `drowsy_swin_training`.
3. **Runtime → Change runtime type → GPU (T4)** → Save.

### B2. Confirm the GPU is active

```python
import torch
print("GPU available:", torch.cuda.is_available())
!nvidia-smi
```

**Functions used:**

| Function | What it does |
|---|---|
| `torch.cuda.is_available()` | Returns `True`/`False` depending on whether PyTorch can see a GPU in this Colab session. Should be `True` after enabling the GPU runtime in B1. |
| `!nvidia-smi` | A terminal command (the `!` runs it as if typed in a terminal, not Python) that prints details about the GPU Colab has assigned — model, memory, usage. |

### B3. Mount Google Drive

Colab's files disappear when the session ends, so anything you want to keep (dataset, checkpoints) must live on Drive.

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/drowsy_project
```

**Functions used:**

| Function | What it does |
|---|---|
| `drive.mount('/content/drive')` | Connects your Google Drive to this Colab session so files can be read from and written to it, under the path `/content/drive/MyDrive/...`. Prompts a one-time permission popup. |
| `!mkdir -p /content/drive/MyDrive/drowsy_project` | Terminal command that creates a folder on Drive (the `-p` means "don't error if it already exists"). |

### B4. Install the model library

PyTorch is pre-installed in Colab; you only need:

```python
!pip install timm
```

### B5. Dataset

1. Download the **MRL Eye** dataset (open/closed eye images). CEW is optional.
2. Upload it to the `drowsy_project` folder on Drive, or load it directly from Kaggle.
3. **Split train/validation/test by person**, not randomly — a random split leaks the same face into both train and test and inflates accuracy.

### B6. Load Swin-T and fine-tune

```python
import torch, timm

device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model("swin_tiny_patch4_window7_224",
                          pretrained=True, num_classes=2)
model.to(device)
```

**Functions used:**

| Function | What it does |
|---|---|
| `timm.create_model(name, pretrained=True, num_classes=2)` | Downloads/loads the named architecture (Swin-T) with weights already trained on millions of general images, and replaces its final output layer with one giving 2 classes (open/closed) instead of its original 1000+ ImageNet classes. |
| `model.to(device)` | Moves the model's numbers onto the GPU (`"cuda"`) if available, so computation happens there instead of the much slower CPU. |
| `torch.utils.data.DataLoader(dataset, batch_size=...)` *(used when building the training loop)* | Feeds the dataset to the model in small batches rather than one image at a time, which trains much faster. |
| `torch.optim.Adam(model.parameters(), lr=...)` *(used when building the training loop)* | The optimizer — the algorithm that adjusts the model's internal numbers after each batch, based on how wrong its predictions were. `lr` (learning rate) controls how big each adjustment step is. |
| `loss_fn(predictions, labels)` e.g. `torch.nn.CrossEntropyLoss()` | Measures how wrong the model's predictions were for a batch, as a single number (the "loss") that the optimizer tries to shrink. |
| `loss.backward()` | Calculates how much each of the model's numbers contributed to the error, so they know which direction to adjust in. |
| `optimizer.step()` | Actually applies that adjustment to the model's numbers. |
| `torch.save(model.state_dict(), "swin_best.pt")` | Saves just the model's learned numbers (not the whole Python object) to a file — this is the "checkpoint" referenced throughout this README. |

- Run a 1-epoch sanity check first to confirm the loss decreases.
- Then train for roughly 5–10 epochs, tracking train/validation loss and accuracy.
- Save the best checkpoint to Drive, e.g. `/content/drive/MyDrive/drowsy_project/swin_best.pt`.

### B7. Evaluate

Compute accuracy, precision, recall, F1, and a confusion matrix on the held-out test set. Save the plots (training curves, confusion matrix) — you'll need them for the report.

### B8. Bring the model back to your laptop

Download the saved checkpoint file from Drive to your `drowsy_project` folder locally. This one file is what Part A5, Step 6 (integration) loads for live inference.

### B9. (Optional) Batch evaluation on recorded test clips

If you'd rather run the EAR-vs-Swin comparison in Colab instead of locally, upload your recorded test clips to Drive and run both methods against them there, producing the results table and charts for the report.

---

## PART C — Swin-T integration (trained checkpoint connected)

A fine-tuned checkpoint (`models/swin_best.pt`, metadata `val_accuracy = 0.9937`) is connected and running locally. The loader reads the class order and normalization from the checkpoint, so its `{OPEN: 0, CLOSED: 1}` mapping is honored instead of the old hardcoded default (which had the classes inverted). EAR remains the default mode; SWIN is selectable and now verified live.

**What is in place:**

| File | Purpose |
|---|---|
| `eye_crop.py` | Extracts left/right eye crops from the existing EAR landmark points. Crops are **square**, 10% padding, centred on the eye and kept inside the frame. Square matters: the classifier trains on square MRL images, and resizing a wide crop to 224x224 squashes the eye and badly hurts live accuracy. |
| `swin_classifier.py` | Reads checkpoint metadata (`model_name`/`arch`, `image_size`, `class_mapping`/`class_names`, `image_mean`/`image_std`), preprocesses RGB/224/ImageNet, and predicts. Uses ONNX Runtime when `models/swin_best.onnx` exists, otherwise a channels-last PyTorch fallback. |
| `export_swin_onnx.py` | Exports the checkpoint to ONNX, compares logits against PyTorch, and benchmarks CPU latency. |
| `drowsy_pipeline.py` | `--mode EAR` (default) or `--mode SWIN`, plus `--swin-threshold`. SWIN replaces only Signal 1 and runs on a background worker so capture stays smooth. EAR and SWIN are never mixed within a session (avoids false alarms while the model warms up). A slow-blink tracker flags repeated 0.4–1.5 s closures (dozing) separately from the >1.5 s microsleep alarm. |
| `test_swin_setup.py` | 15 checks: preprocessing, metadata adoption, square crops, graceful failures, real inference, background-worker sequencing, and slow-blink tracking. |
| `docs/swin_training_contract.md` | Training/inference contract, metadata keys, and what a retrained checkpoint must match. |

**Commands:**

```powershell
# EAR mode
python drowsy_pipeline.py

# SWIN mode (checkpoint and ONNX export auto-detected)
uv sync --extra ml
python drowsy_pipeline.py --mode SWIN
python drowsy_pipeline.py --mode SWIN --swin-threshold 0.55

# Diagnostics / re-export
python test_swin_setup.py
python swin_classifier.py
python export_swin_onnx.py
```

On this laptop (CPU only) ONNX Runtime takes ~0.24 s per eye with 2 threads versus ~0.73 s with PyTorch, so the background worker refreshes the SWIN score about twice per second while the camera loop keeps running at full speed. Live check on 2026-11-18: open eyes scored 0.000; sustained closure produced `ALARM: sustained eye closure`, and reopening produced `ALARM cleared: awake`.

**Slow-blink / dozing detection:** normal blinks (<0.4 s) are ignored; closures of 0.4–1.5 s are counted as slow blinks when at least two consecutive fresh measurements agree (so the ~2 Hz SWIN update cannot turn one normal blink into a slow one). Three slow blinks within 60 s alarm as dozing, and the alarm clears 20 s after the last one. Defaults are at the top of `drowsy_pipeline.py` (`SLOW_BLINK_*`); they are heuristic until calibrated on your own clips.

---

## 3. Known risks / things to check early

- **CPU speed for live Swin-T inference.** Mitigated here with ONNX Runtime plus a background worker (~2 score updates/s, camera loop unaffected). Measure on any new laptop; if slower, try INT8 quantization or a lighter backbone (MobileNetV3/EfficientNet-B0).
- **Dataset split leakage.** Always split by person, never randomly.
- **Lighting and glasses.** The system is built for a standard webcam in reasonable lighting; night driving and heavy occlusion (sunglasses) are explicitly out of scope — state this in the report, don't claim it works.
- **MediaPipe version.** This project uses the newer MediaPipe Tasks API (`FaceLandmarker`), which differs from the older `mp.solutions.face_mesh` seen in many tutorials. If following an outdated tutorial, expect API differences.
- **Three signals means three sets of false alarms to check.** Talking can look like yawning if the minimum-duration rule is too short; normal activity (looking down to change gears, checking mirrors) can look like PERCLOS creep if the rolling window or threshold is too tight. The slow-blink/dozing defaults are also heuristic until calibrated. Test each signal individually before relying on the combined fusion rule, and log which signal fired on every alarm during evaluation so you can tell them apart.
- **Report the fusion rule's net effect, not just each signal alone.** Combining three signals with an "any fires" rule usually catches more real drowsiness but can also raise the total false-alarm rate, since any one signal firing is enough. State both numbers (per-signal and combined) in the results.

---

## 4. Final deliverables checklist

- [ ] Working live demo on a laptop (EAR method at minimum)
- [ ] Trained Swin-T checkpoint
- [ ] EAR vs Swin results table and charts
- [ ] Source code on GitHub with this README
- [ ] Final report
- [ ] Final presentation
- [ ] Backup demo video