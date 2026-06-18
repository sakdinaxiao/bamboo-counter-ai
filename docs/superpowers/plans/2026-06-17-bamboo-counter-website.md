# Bamboo Counter AI — Website Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a locally-hosted web UI where a user uploads a video and sees the bamboo count returned by the existing ML pipeline.

**Architecture:** FastAPI serves a single `index.html` at `GET /` and accepts video uploads at `POST /count`. A headless wrapper (`web_main.py`) re-implements the inference loop from `src/main.py` without any `cv2` display calls and with a graceful fallback when the segmentation model is absent. All frontend state (upload → processing → result) is managed in vanilla JS with no build step.

**Tech Stack:** Python 3.x, FastAPI, uvicorn, python-multipart, existing PyTorch/YOLO/supervision stack.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `web_main.py` | Headless inference wrapper — no display, segmentation fallback, returns `int` count |
| Create | `app.py` | FastAPI app — serves HTML, accepts upload, runs inference in thread pool |
| Create | `templates/index.html` | Single-page UI — upload / processing / result states, bamboo green theme |
| Create | `tests/test_app.py` | API-layer tests using FastAPI TestClient |
| Modify | `requirements.txt` | Add `fastapi`, `uvicorn[standard]`, `python-multipart`, `httpx` |

---

## Task 1: Install dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add web dependencies to requirements.txt**

Append these four lines to the end of `requirements.txt`:

```
fastapi==0.115.12
uvicorn[standard]==0.34.3
python-multipart==0.0.20
httpx==0.28.1
```

- [ ] **Step 2: Install them**

```bash
pip install fastapi "uvicorn[standard]" python-multipart httpx
```

Expected: all four packages install without error.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add fastapi, uvicorn, python-multipart, httpx"
```

---

## Task 2: Create headless inference wrapper

**Files:**
- Create: `web_main.py`
- Create: `tests/__init__.py`
- Create: `tests/test_app.py` (stub only — full tests in Task 4)

`web_main.py` re-implements the `src/main.py` inference loop with three changes:
1. All `cv2.imshow` / `cv2.waitKey` / `cv2.destroyAllWindows` calls are removed.
2. If `training_result/segment/best.pt` does not exist, it skips segmentation and treats the full frame as a single region.
3. Raises `FileNotFoundError` if the detection model is missing (instead of silently returning `None`).

- [ ] **Step 1: Create `web_main.py`**

```python
from pathlib import Path
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from src.sahi_inference import apply_sahi, get_sahi
from src.clahe_inference import apply_clahe
from src.global_tracker import GlobalTracker

project_root = Path(__file__).resolve().parent
_model_path = project_root / "training_result" / "detection_small" / "weights" / "best.pt"
_seg_model_path = project_root / "training_result" / "segment" / "best.pt"


def run(video_path: str) -> int:
    if not _model_path.exists():
        raise FileNotFoundError(f"Detection model not found: {_model_path}")

    use_segmentation = _seg_model_path.exists()
    seg_model = YOLO(str(_seg_model_path)) if use_segmentation else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        sahi = get_sahi(_model_path)
        tracker = GlobalTracker(merge_distance=12.5)

        origin_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        stride = max(1, int(origin_fps / 3))
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % stride != 0:
                continue

            if use_segmentation:
                from src.segmentation import segmenting
                regions = segmenting(seg_model, frame)
            else:
                regions = [{"image": frame, "offset_x": 0, "offset_y": 0}]

            frame = apply_clahe(frame=frame)
            all_xyxy, all_conf, all_cls = [], [], []

            for region in regions:
                reg_img = apply_clahe(region["image"])
                detected = apply_sahi(sahi, reg_img)
                ox, oy = region["offset_x"], region["offset_y"]

                if not detected.is_empty():
                    for i in range(len(detected.xyxy)):
                        all_xyxy.append([
                            detected.xyxy[i][0] + ox,
                            detected.xyxy[i][1] + oy,
                            detected.xyxy[i][2] + ox,
                            detected.xyxy[i][3] + oy,
                        ])
                        all_conf.append(detected.confidence[i])
                        all_cls.append(detected.class_id[i])

            if all_xyxy:
                detections = sv.Detections(
                    xyxy=np.array(all_xyxy),
                    confidence=np.array(all_conf),
                    class_id=np.array(all_cls),
                )
            else:
                detections = sv.Detections.empty()

            tracker.update(frame, detections)

        return tracker.count()
    finally:
        cap.release()
```

- [ ] **Step 2: Create `tests/__init__.py`**

```python
```

(Empty file — just makes `tests/` a package.)

- [ ] **Step 3: Commit**

```bash
git add web_main.py tests/__init__.py
git commit -m "feat: add headless inference wrapper web_main.py"
```

---

## Task 3: Create FastAPI server

**Files:**
- Create: `app.py`
- Create: `templates/` directory (empty, filled in Task 4)

- [ ] **Step 1: Create `templates/` directory**

```bash
mkdir templates
```

- [ ] **Step 2: Create `app.py`**

```python
import asyncio
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

import web_main

app = FastAPI()

_templates = Path(__file__).parent / "templates"
_temp_dir = Path(__file__).parent / "temp"
_temp_dir.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = _templates / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=503, detail="Frontend not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/count")
async def count_bamboo(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    suffix = Path(file.filename or "upload").suffix or ".mp4"
    temp_path = _temp_dir / f"{uuid.uuid4()}{suffix}"

    try:
        temp_path.write_bytes(await file.read())
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(None, web_main.run, str(temp_path))
        return JSONResponse({"count": count})
    except Exception as exc:
        return JSONResponse({"count": 0, "error": str(exc)})
    finally:
        if temp_path.exists():
            temp_path.unlink()
```

- [ ] **Step 3: Verify the server starts (no templates yet — expect 503 on GET /)**

```bash
uvicorn app:app --reload
```

Open a second terminal and run:

```bash
curl http://localhost:8000/
```

Expected: `{"detail":"Frontend not found"}` with status 503. This confirms FastAPI is running.

Stop the server (`Ctrl+C`).

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add FastAPI server with upload endpoint"
```

---

## Task 4: Write API tests

**Files:**
- Modify: `tests/test_app.py`

These tests use FastAPI's `TestClient` (backed by `httpx`) and mock `web_main.run` so no ML models are needed.

- [ ] **Step 1: Write `tests/test_app.py`**

```python
from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_get_index_returns_html():
    # templates/index.html must exist for this test
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_count_rejects_non_video():
    response = client.post(
        "/count",
        files={"file": ("photo.jpg", BytesIO(b"fake"), "image/jpeg")},
    )
    assert response.status_code == 400
    assert "video" in response.json()["detail"].lower()


def test_count_returns_count_on_success():
    with patch("app.web_main.run", return_value=42):
        response = client.post(
            "/count",
            files={"file": ("clip.mp4", BytesIO(b"fake video bytes"), "video/mp4")},
        )
    assert response.status_code == 200
    assert response.json() == {"count": 42}


def test_count_returns_zero_and_error_on_failure():
    with patch("app.web_main.run", side_effect=FileNotFoundError("model missing")):
        response = client.post(
            "/count",
            files={"file": ("clip.mp4", BytesIO(b"fake video bytes"), "video/mp4")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert "model missing" in data["error"]
```

- [ ] **Step 2: Run tests — expect `test_get_index_returns_html` to FAIL (no HTML yet), rest to PASS**

```bash
pytest tests/test_app.py -v
```

Expected output:
```
FAILED tests/test_app.py::test_get_index_returns_html - AssertionError
PASSED tests/test_app.py::test_count_rejects_non_video
PASSED tests/test_app.py::test_count_returns_count_on_success
PASSED tests/test_app.py::test_count_returns_zero_and_error_on_failure
```

- [ ] **Step 3: Commit tests**

```bash
git add tests/test_app.py
git commit -m "test: add FastAPI endpoint tests"
```

---

## Task 5: Create frontend HTML

**Files:**
- Create: `templates/index.html`

- [ ] **Step 1: Create `templates/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Bamboo Counter AI</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #0A0A0A;
      --bg-surface: #141414;
      --accent: #16A34A;
      --accent-hover: #22C55E;
      --text-primary: #FFFFFF;
      --text-muted: #B3B3B3;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Inter', 'Helvetica Neue', sans-serif;
      background: var(--bg-base);
      color: var(--text-primary);
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* ── Hero ── */
    .hero {
      height: 75vh;
      min-height: 480px;
      background: linear-gradient(135deg, #0a1f0a 0%, #0A0A0A 55%, #0d1a0d 100%);
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 0 2rem;
    }
    .hero::after {
      content: '';
      position: absolute;
      bottom: 0; left: 0; right: 0;
      height: 55%;
      background: linear-gradient(to top, var(--bg-surface), transparent);
      pointer-events: none;
    }
    .hero-content { position: relative; z-index: 1; }
    .hero-badge {
      display: inline-block;
      background: rgba(22,163,74,0.12);
      border: 1px solid rgba(22,163,74,0.35);
      color: var(--accent-hover);
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      padding: 0.35rem 1rem;
      border-radius: 2rem;
      margin-bottom: 1.5rem;
    }
    .hero-title {
      font-size: clamp(2.6rem, 6vw, 5.2rem);
      font-weight: 800;
      letter-spacing: -0.02em;
      line-height: 1.05;
      margin-bottom: 1rem;
    }
    .hero-title span { color: var(--accent-hover); }
    .hero-subtitle {
      font-size: clamp(0.95rem, 2vw, 1.2rem);
      color: var(--text-muted);
      max-width: 540px;
      margin: 0 auto 2.5rem;
      line-height: 1.65;
    }
    .btn-primary {
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 0.9rem 2.5rem;
      font-size: 1rem;
      font-weight: 700;
      border-radius: 4px;
      cursor: pointer;
      transition: background 300ms ease-in-out, transform 150ms ease;
      letter-spacing: 0.02em;
    }
    .btn-primary:hover { background: var(--accent-hover); transform: scale(1.03); }

    /* ── Content ── */
    .content {
      background: var(--bg-surface);
      min-height: 30vh;
      padding: 4rem 2rem 7rem;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    /* ── Upload zone ── */
    .upload-zone {
      width: 100%;
      max-width: 580px;
      border: 2px dashed rgba(22,163,74,0.35);
      border-radius: 12px;
      padding: 4rem 2rem;
      text-align: center;
      cursor: pointer;
      transition: border-color 300ms ease-in-out, transform 300ms ease-in-out, box-shadow 300ms ease-in-out;
      background: rgba(22,163,74,0.03);
    }
    .upload-zone:hover, .upload-zone.drag-over {
      border-color: var(--accent-hover);
      transform: scale(1.03);
      box-shadow: 0 0 48px rgba(34,197,94,0.12);
    }
    .upload-icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.65; }
    .upload-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; }
    .upload-sub { color: var(--text-muted); font-size: 0.88rem; margin-bottom: 1.75rem; }
    #file-input { display: none; }

    /* ── Processing ── */
    .processing { display: none; text-align: center; padding: 3rem 2rem; }
    .spinner {
      width: 60px; height: 60px;
      border: 3px solid rgba(22,163,74,0.18);
      border-top-color: var(--accent-hover);
      border-radius: 50%;
      animation: spin 0.85s linear infinite;
      margin: 0 auto 2rem;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .processing-label { font-size: 1.05rem; color: var(--text-muted); }

    /* ── Result ── */
    .result { display: none; text-align: center; padding: 2rem; }
    .result-count {
      font-size: clamp(5rem, 14vw, 10rem);
      font-weight: 800;
      letter-spacing: -0.04em;
      color: var(--accent-hover);
      text-shadow: 0 0 80px rgba(34,197,94,0.45), 0 0 28px rgba(34,197,94,0.25);
      line-height: 1;
      margin-bottom: 0.5rem;
    }
    .result-label { font-size: 1.2rem; color: var(--text-muted); margin-bottom: 2.5rem; }
    .result-error { color: #f87171; font-size: 0.88rem; margin-bottom: 1.5rem; }
    .btn-secondary {
      background: transparent;
      color: var(--text-primary);
      border: 2px solid rgba(255,255,255,0.18);
      padding: 0.75rem 2rem;
      font-size: 0.95rem;
      font-weight: 600;
      border-radius: 4px;
      cursor: pointer;
      transition: border-color 300ms ease, background 300ms ease;
    }
    .btn-secondary:hover {
      border-color: var(--accent);
      background: rgba(22,163,74,0.1);
    }
  </style>
</head>
<body>

<section class="hero">
  <div class="hero-content">
    <div class="hero-badge">AI-Powered Vision</div>
    <h1 class="hero-title">Bamboo<br><span>Counter AI</span></h1>
    <p class="hero-subtitle">Upload a drone or camera video and our AI will detect, track, and count every unique bamboo stalk — automatically.</p>
    <button class="btn-primary" id="hero-btn">Upload Video</button>
  </div>
</section>

<section class="content">
  <div class="upload-zone" id="upload-zone">
    <div class="upload-icon">🎬</div>
    <div class="upload-title">Drop your video here</div>
    <div class="upload-sub">or click to browse · MP4, MOV, AVI accepted</div>
    <button class="btn-primary" id="browse-btn">Choose File</button>
    <input type="file" id="file-input" accept="video/*">
  </div>

  <div class="processing" id="processing">
    <div class="spinner"></div>
    <div class="processing-label" id="processing-label">Loading model…</div>
  </div>

  <div class="result" id="result">
    <div class="result-count" id="result-count">0</div>
    <div class="result-label">bamboo detected in your video</div>
    <div class="result-error" id="result-error" style="display:none"></div>
    <button class="btn-secondary" id="reset-btn">Count another video</button>
  </div>
</section>

<script>
  const uploadZone   = document.getElementById('upload-zone');
  const processingEl = document.getElementById('processing');
  const resultEl     = document.getElementById('result');
  const fileInput    = document.getElementById('file-input');
  const labelEl      = document.getElementById('processing-label');
  const countEl      = document.getElementById('result-count');
  const errorEl      = document.getElementById('result-error');

  const STEPS = ['Loading model…', 'Analyzing frames…', 'Counting bamboo…'];
  let stepTimer = null;

  document.getElementById('hero-btn').addEventListener('click', () => fileInput.click());
  document.getElementById('browse-btn').addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
  document.getElementById('reset-btn').addEventListener('click', resetToUpload);

  uploadZone.addEventListener('click', () => fileInput.click());
  uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  });
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

  function handleFile(file) {
    if (!file.type.startsWith('video/')) {
      alert('Please upload a video file (MP4, MOV, AVI, etc.).');
      return;
    }
    showProcessing();
    uploadVideo(file);
  }

  function showProcessing() {
    uploadZone.style.display = 'none';
    resultEl.style.display   = 'none';
    processingEl.style.display = 'block';
    let i = 0;
    labelEl.textContent = STEPS[0];
    stepTimer = setInterval(() => {
      i = (i + 1) % STEPS.length;
      labelEl.textContent = STEPS[i];
    }, 3000);
  }

  function showResult(count, error) {
    clearInterval(stepTimer);
    processingEl.style.display = 'none';
    resultEl.style.display = 'block';
    if (error) {
      errorEl.textContent = error;
      errorEl.style.display = 'block';
      countEl.textContent = '0';
    } else {
      errorEl.style.display = 'none';
      animateCount(count);
    }
  }

  function animateCount(target) {
    const duration = 1500;
    const t0 = performance.now();
    function tick(now) {
      const p = Math.min((now - t0) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      countEl.textContent = Math.round(eased * target);
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  function resetToUpload() {
    resultEl.style.display     = 'none';
    processingEl.style.display = 'none';
    uploadZone.style.display   = 'block';
    fileInput.value = '';
  }

  async function uploadVideo(file) {
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res  = await fetch('/count', { method: 'POST', body: fd });
      const data = await res.json();
      showResult(data.count, data.error || null);
    } catch {
      showResult(0, 'Network error — is the server running?');
    }
  }
</script>
</body>
</html>
```

- [ ] **Step 2: Run all tests — all four should now PASS**

```bash
pytest tests/test_app.py -v
```

Expected:
```
PASSED tests/test_app.py::test_get_index_returns_html
PASSED tests/test_app.py::test_count_rejects_non_video
PASSED tests/test_app.py::test_count_returns_count_on_success
PASSED tests/test_app.py::test_count_returns_zero_and_error_on_failure
```

- [ ] **Step 3: Start the server and manually verify the UI**

```bash
uvicorn app:app --reload
```

Open `http://localhost:8000` in a browser. Verify:
- Hero section fills ~75% of viewport height
- "Upload Video" button and upload zone are visible below
- Hovering the upload zone shows green glow and slight scale
- Dragging a file over the zone highlights the border
- Selecting a non-video file shows the alert

- [ ] **Step 4: Commit**

```bash
git add templates/index.html
git commit -m "feat: add Netflix-inspired bamboo counter frontend"
```

---

## Task 6: End-to-end smoke test with a real video

> Skip this task if no video file is available. The ML models may take several minutes on CPU.

**Files:** no code changes

- [ ] **Step 1: Place a test video in the project root**

Copy any `.mp4` or `.mov` file into `C:\Users\wonbi\pythoncode\NectecBam\`.

- [ ] **Step 2: Start the server**

```bash
uvicorn app:app --reload
```

- [ ] **Step 3: Open `http://localhost:8000` and upload the video**

Observe:
- Processing spinner appears immediately after file selection
- Step labels cycle through "Loading model…" → "Analyzing frames…" → "Counting bamboo…"
- After inference completes, the count animates from 0 to N with a green glow
- "Count another video" button resets the UI to the upload state

- [ ] **Step 4: Check server logs for any errors**

Expected: no Python tracebacks. If the segmentation model is missing, the log should show the fallback message but still return a count.

---

## How to run

```bash
# From the project root:
uvicorn app:app --reload
# Open: http://localhost:8000
```
