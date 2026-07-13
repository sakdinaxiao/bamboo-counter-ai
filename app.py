import asyncio
import functools
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import web_main

_templates = Path(__file__).parent / "templates"
_temp_dir = Path(__file__).parent / "temp"
_temp_dir.mkdir(exist_ok=True)

_VIDEO_TTL_SECONDS = 600  # 10 minutes

_progress: dict[str, int] = {}


async def _cleanup_old_videos():
    while True:
        await asyncio.sleep(_VIDEO_TTL_SECONDS)
        cutoff = time.time() - _VIDEO_TTL_SECONDS
        for f in _temp_dir.glob("*_output.mp4"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
            except OSError:
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_cleanup_old_videos())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = _templates / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=503, detail="Frontend not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/count")
async def count_bamboo(file: UploadFile = File(...), job_id: str | None = Form(None)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    if job_id is not None:
        try:
            uuid.UUID(job_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid job ID")
        jid = job_id
    else:
        jid = str(uuid.uuid4())

    suffix = Path(file.filename or "upload").suffix or ".mp4"
    temp_path = _temp_dir / f"{jid}_input{suffix}"
    output_path = _temp_dir / f"{jid}_output.mp4"

    def _report(pct: int) -> None:
        _progress[jid] = pct

    try:
        temp_path.write_bytes(await file.read())
        loop = asyncio.get_running_loop()
        count = await loop.run_in_executor(
            None,
            functools.partial(
                web_main.run, str(temp_path), str(output_path), progress=_report
            ),
        )
        response: dict = {"count": count}
        if output_path.exists():
            response["video_id"] = jid
        return JSONResponse(response)
    except Exception as exc:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        return JSONResponse({"count": 0, "error": str(exc)})
    finally:
        temp_path.unlink(missing_ok=True)
        _progress.pop(jid, None)


@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID")
    return JSONResponse({"percent": _progress.get(job_id, 0)})


@app.get("/video/{video_id}")
async def get_video(video_id: str):
    try:
        uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID")

    video_path = _temp_dir / f"{video_id}_output.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(str(video_path), media_type="video/mp4")


@app.delete("/video/{video_id}", status_code=204)
async def delete_video(video_id: str):
    try:
        uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID")

    video_path = _temp_dir / f"{video_id}_output.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    video_path.unlink(missing_ok=True)
