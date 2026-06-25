import asyncio
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

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
    job_id = uuid.uuid4()
    temp_path = _temp_dir / f"{job_id}_input{suffix}"
    output_path = _temp_dir / f"{job_id}_output.mp4"

    try:
        temp_path.write_bytes(await file.read())
        loop = asyncio.get_running_loop()
        count = await loop.run_in_executor(
            None, web_main.run, str(temp_path), str(output_path)
        )
        response: dict = {"count": count}
        if output_path.exists():
            response["video_id"] = str(job_id)
        return JSONResponse(response)
    except Exception as exc:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        return JSONResponse({"count": 0, "error": str(exc)})
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.get("/video/{video_id}")
async def get_video(video_id: str, background_tasks: BackgroundTasks):
    try:
        uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID")

    video_path = _temp_dir / f"{video_id}_output.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    background_tasks.add_task(video_path.unlink, missing_ok=True)
    return FileResponse(str(video_path), media_type="video/mp4")
