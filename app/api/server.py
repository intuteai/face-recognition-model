import logging
import time
import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request

from app.logging_config import setup_logging
from app.core.recognize import Pipeline

# ── Setup logging & app ─────────────────────────────────────────
setup_logging()
app = FastAPI(title="Attendance Face API")
log = logging.getLogger(__name__)
PIPE = Pipeline()

# ── HTTP request/response logging middleware ───────────────────
@app.middleware("http")
async def http_logger(request: Request, call_next):
    start = time.perf_counter()
    path = request.url.path
    method = request.method
    client = request.client.host if request.client else "unknown"
    try:
        response = await call_next(request)
        dt = int((time.perf_counter() - start) * 1000)
        logging.getLogger("app.http").info(
            "http",
            extra={
                "method": method,
                "path": path,
                "status": response.status_code,
                "client": client,
                "ms": dt,
            },
        )
        return response
    except Exception as e:
        dt = int((time.perf_counter() - start) * 1000)
        logging.getLogger("app.http").exception(
            "http_error",
            extra={"method": method, "path": path, "client": client, "ms": dt, "error": str(e)},
        )
        raise

# ── Always return JSON for errors ───────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": "validation_error", "details": exc.errors()})

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logging.getLogger("app.api.server").exception("unhandled_exception")
    return JSONResponse(status_code=500, content={"error": str(exc)})

# ── Helpers ────────────────────────────────────────────────────
def _read_img_from_upload(file: UploadFile) -> np.ndarray:
    raw = file.file.read()
    if not raw:
        raise ValueError(f"Empty image upload: {getattr(file, 'filename', None)}")
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Invalid image upload: {getattr(file, 'filename', None)} bytes={len(raw)}")
    return img

# ── Endpoints ──────────────────────────────────────────────────
@app.get("/healthz")
def healthz():
    return {"status": "ok", "people": PIPE.db.people()}

@app.post("/enroll")
async def enroll(person_id: str = Query(...), files: list[UploadFile] = File(...)):
    log.info("api_enroll_in", extra={"person_id": person_id, "files": [getattr(f, "filename", None) for f in files]})
    total_faces = total_emb = 0
    for f in files:
        try:
            raw = f.file.read()
            log.info(
                "upload_info",
                extra={
                    "upload_filename": getattr(f, "filename", None),
                    "size_bytes": len(raw) if raw else 0,
                    "content_type": getattr(f, "content_type", None),
                },
            )
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Invalid image upload {getattr(f, 'filename', None)}")
            res = PIPE.enroll_image(img, person_id, getattr(f, "filename", None) or "upload")
            total_faces += res["faces"]
            total_emb += res["embeddings_added"]
        except Exception as e:
            log.exception("enroll_failed", extra={"file": getattr(f, "filename", None)})
            return JSONResponse(status_code=400, content={"error": str(e)})
    log.info("api_enroll_out", extra={"person_id": person_id, "faces": total_faces, "embeddings": total_emb})
    return {"person_id": person_id, "faces": total_faces, "embeddings": total_emb}

@app.post("/enroll/video")
async def enroll_video(person_id: str = Query(...), video: UploadFile = File(...)):
    try:
        raw = video.file.read()
        if not raw:
            return JSONResponse(status_code=400, content={"error": "Empty video upload"})
        log.info(
            "api_video_in",
            extra={
                "person_id": person_id,
                "upload_filename": getattr(video, "filename", None),
                "size_bytes": len(raw),
                "content_type": getattr(video, "content_type", None),
            },
        )
        res = PIPE.enroll_video(raw, person_id=person_id, source=(getattr(video, "filename", None) or "video"))
        log.info("api_video_out", extra={"person_id": person_id, **res})
        return {"person_id": person_id, **res}
    except Exception as e:
        log.exception("enroll_video_failed", extra={"file": getattr(video, "filename", None)})
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        raw = file.file.read()
        log.info(
            "api_rec_in",
            extra={
                "upload_filename": getattr(file, "filename", None),
                "size_bytes": len(raw) if raw else 0,
                "content_type": getattr(file, "content_type", None),
            },
        )
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Invalid image upload {getattr(file, 'filename', None)}")
        preds = PIPE.recognize_image(img)
        log.info("api_rec_out", extra={"faces": len(preds)})
        return {"faces": preds}
    except Exception as e:
        log.exception("recognize_failed", extra={"file": getattr(file, "filename", None)})
        return JSONResponse(status_code=400, content={"error": str(e)})

# (Optional) Friendly root
@app.get("/")
def root():
    return {"message": "Attendance Face API", "try": ["/healthz", "/docs", "POST /enroll", "POST /enroll/video", "POST /recognize"]}
