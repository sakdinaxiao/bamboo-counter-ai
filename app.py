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
        loop = asyncio.get_running_loop()
        count = await loop.run_in_executor(None, web_main.run, str(temp_path))
        return JSONResponse({"count": count})
    except Exception as exc:
        return JSONResponse({"count": 0, "error": str(exc)})
    finally:
        if temp_path.exists():
            temp_path.unlink()
