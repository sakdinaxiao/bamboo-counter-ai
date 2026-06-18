# Bamboo Counter AI — Website Design Spec
**Date:** 2026-06-17  
**Status:** Approved

---

## Goal

A locally-hosted single-page website that lets a user upload a video, runs the existing bamboo-counting ML pipeline, and displays the final count. Simple: one action in, one number out.

---

## Architecture

### Files added to the repo

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server — video upload endpoint + static file serving |
| `web_main.py` | Headless wrapper around existing `src/main.py` |
| `templates/index.html` | Single-page frontend (CSS + JS inline) |

No new dependencies beyond `fastapi` and `uvicorn[standard]` (added to `requirements.txt`).

### How to run

```bash
pip install fastapi "uvicorn[standard]"
uvicorn app:app --reload
# → open http://localhost:8000
```

---

## Backend

### `web_main.py`

Wraps `src/main.py` with two changes:
1. Removes all `cv2.imshow`, `cv2.waitKey`, `cv2.destroyAllWindows` calls so inference runs headless (no display required).
2. If the segmentation model (`training_result/segment/best.pt`) is missing, skips the segmentation step and processes the full frame directly — returns count instead of exiting early.

Signature:
```python
def run(video_path: str) -> int:
    ...
    return count
```

### `app.py`

```
GET  /              → serves templates/index.html
POST /count         → multipart video upload → runs web_main.run() → { "count": N }
```

- Saves uploaded video to `temp/<uuid>.<ext>` before processing.
- Runs blocking inference in `asyncio` thread pool (`run_in_executor`) so the server stays responsive.
- Deletes temp file in a `finally` block regardless of outcome.
- Returns `{ "count": 0, "error": "..." }` on failure (never 500s the client).

---

## Frontend — `templates/index.html`

Single HTML file. All CSS and JS are inline (no external build tools or CDN dependencies beyond a Google Font import).

### Design tokens

| Token | Value |
|-------|-------|
| Background base | `#0A0A0A` |
| Background surface | `#141414` |
| Accent primary | `#16A34A` (bamboo green) |
| Accent hover | `#22C55E` |
| Text primary | `#FFFFFF` |
| Text muted | `#B3B3B3` |
| Font | Inter (Google Fonts) → fallback Helvetica Neue, sans-serif |

### Layout — three states rendered in the same viewport

**State 1: Upload**
- Hero section: `70vh`, title "Bamboo Counter AI", subtitle, green CTA button "Upload Video"
- Bottom gradient: `linear-gradient(to top, #141414, transparent 50%)` blending hero into surface
- Upload zone: centered card, dashed `#16A34A` border, drag-and-drop target + file picker fallback
- Accepted formats: `video/*`

**State 2: Processing**
- Replaces upload zone in place (no page navigation)
- Spinner animation (CSS `@keyframes` rotation) in accent green
- Step label cycles through: "Loading model…" → "Analyzing frames…" → "Counting bamboo…" every 3s
- No cancel button (processing is synchronous; page reload cancels)

**State 3: Result**
- Large counter number animates from 0 to N over 1.5s (JS `requestAnimationFrame` with ease-out)
- Green glow on the number via `text-shadow`
- Subtext: "bamboo detected in your video"
- "Count another video" button resets to State 1

### Micro-interactions

| Interaction | Behavior |
|-------------|----------|
| Drag file over upload zone | `scale(1.03)` + brighter green border, 300ms ease-in-out |
| Hover CTA / reset button | Background `#16A34A` → `#22C55E`, 300ms ease |
| Count reveal | JS counter 0 → N, 1.5s, ease-out easing function |

---

## Error handling

| Scenario | Behavior |
|----------|----------|
| No segmentation model | `web_main.py` falls back to full-frame processing; still returns a count |
| Invalid file type | Frontend validates `file.type.startsWith('video/')` before upload; shows inline error |
| Backend inference error | `POST /count` returns `{ "count": 0, "error": "..." }`; frontend shows error message in result state |
| Temp file cleanup | `finally` block in endpoint always deletes temp file |

---

## Out of scope

- Authentication or user sessions
- Upload history or result persistence
- Progress streaming (SSE/WebSocket) — synchronous response is sufficient for local use
- Mobile layout optimization
- Deployment to cloud (local only)
