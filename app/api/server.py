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

from fastapi import Form

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

@app.post("/attendance/mark")
async def attendance_mark(
    file: UploadFile = File(...),
    confirm_user_id: str | None = Form(None),  # optional override from app
):
    """
    One-shot attendance from a single frame.

    - If confirm_user_id is provided, we mark attendance for that user_id directly.
    - Else we run recognition:
        * similarity >= ATTEND_AUTO_THRESHOLD  -> auto mark
        * ATTEND_MAYBE_THRESHOLD..AUTO         -> ask client to confirm (return candidates)
        * below MAYBE or unknown               -> no_match
    Returns JSON with status and details so app knows who was marked (or to confirm).
    """
    from app.clients.http import post_json
    from app import config

    try:
        raw = file.file.read()
        if not raw:
            return JSONResponse(status_code=400, content={"error": "Empty image upload"})

        log.info(
            "att_mark_in",
            extra={
                "upload_filename": getattr(file, "filename", None),
                "bytes": len(raw),
                "content_type": getattr(file, "content_type", None),
                "confirm_user_id": confirm_user_id,
            },
        )

        # Decode
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image"})

        # Helper to convert any local person_id to ERP user_id (int), using map if needed
        def to_erp_user_id(pid: str) -> int | None:
            try:
                return int(pid)
            except Exception:
                return getattr(config, "ERP_USER_MAP", {}).get(pid)

        # If client already confirmed a user_id, mark straight away
        if confirm_user_id:
            erp_uid = to_erp_user_id(confirm_user_id)
            if erp_uid is None:
                return {"status": "error", "message": f"Invalid or unmapped confirm_user_id: {confirm_user_id}"}
            payload = {"user_id": erp_uid}
            status, resp = await post_json(config.ERP_ATTENDANCE_URL, payload)
            log.info("att_mark_confirmed", extra={"confirm_user_id": confirm_user_id, "erp_uid": erp_uid, "status": status, "resp": resp})
            return {
                "status": "marked",
                "chosen_user_id": confirm_user_id,
                "erp_user_id": erp_uid,
                "attendance": resp,
                "message": "Attendance marked via confirmation."
            }

        # Run recognition
        preds = PIPE.recognize_image(img)  # your existing pipeline
        if not preds:
            return {"status": "no_face", "message": "No face detected."}

        best = preds[0]  # top detection by score
        best_pid = best["prediction"]["person_id"]
        best_sim = float(best["prediction"]["similarity"])
        top_k = best.get("top_k", [])  # list of {person_id, score}

        auto_thr  = float(getattr(config, "ATTEND_AUTO_THRESHOLD", 0.75))
        maybe_thr = float(getattr(config, "ATTEND_MAYBE_THRESHOLD", 0.60))

        # Unknown or very low similarity → no match
        if best_pid == "unknown" or best_sim < maybe_thr:
            return {
                "status": "no_match",
                "best": {"person_id": best_pid, "similarity": best_sim},
                "candidates": top_k[: int(getattr(config, "TOP_K", 3))],
                "message": "Face not confidently recognized.",
            }

        # High confidence → auto mark attendance
        if best_sim >= auto_thr:
            erp_uid = to_erp_user_id(best_pid)
            if erp_uid is None:
                # We recognized locally but can't map to ERP → ask client to confirm / map
                return {
                    "status": "needs_confirmation",
                    "best": {"person_id": best_pid, "similarity": best_sim},
                    "candidates": top_k[: int(getattr(config, "TOP_K", 3))],
                    "message": f"ERP mapping missing for '{best_pid}'. Provide confirm_user_id to mark.",
                }
            payload = {"user_id": erp_uid}
            status, resp = await post_json(config.ERP_ATTENDANCE_URL, payload)
            log.info("att_mark_auto", extra={"person_id": best_pid, "erp_uid": erp_uid, "similarity": best_sim, "status": status, "resp": resp})
            return {
                "status": "marked",
                "best": {"person_id": best_pid, "similarity": best_sim},
                "erp_user_id": erp_uid,
                "attendance": resp,
                "message": "Attendance marked automatically.",
            }

        # Medium confidence → require client confirmation
        return {
            "status": "needs_confirmation",
            "best": {"person_id": best_pid, "similarity": best_sim},
            "candidates": top_k[: int(getattr(config, "TOP_K", 3))],
            "message": "Please confirm the correct person_id.",
        }

    except Exception as e:
        log.exception("att_mark_failed", extra={"error": str(e)})
        return JSONResponse(status_code=400, content={"error": str(e)})
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
from fastapi import Form  # add at top with other imports

@app.post("/register")
async def register(
    name: str = Form(""),
    user_id: str = Form(...),
    files: list[UploadFile] = File(...),
    max_images: int = Query(20, ge=1, le=200),  # keep in sync with app
):
    """
    Receive pre-cut frames from the app, preprocess (detect+align), embed, and store.
    - person_id is set to user_id (string safe); if numeric, you can map later to ERP.
    - returns a summary with how many faces/embeddings were created.
    """
    from app import config

    person_id = str(user_id).strip()
    if not person_id:
        return JSONResponse(status_code=400, content={"error": "user_id required"})

    if not files:
        return JSONResponse(status_code=400, content={"error": "No files[] received"})

    # hard cap to protect server; app already caps at 20
    cap = min(max_images, getattr(config, "MAX_IMAGES", max_images))

    log.info(
        "api_register_in",
        extra={
            "person_id": person_id,
            "person_name": name,
            "files_count": len(files),
            "cap": cap,
        },
    )

    total_faces = 0
    total_emb = 0
    processed = 0
    errors = 0

    for i, f in enumerate(files[:cap], start=1):
        try:
            raw = f.file.read()
            if not raw:
                errors += 1
                log.warning("register_empty_file", extra={"idx": i, "upload_filename": getattr(f, "filename", None)})
                continue

            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                errors += 1
                log.warning("register_decode_fail", extra={"idx": i, "upload_filename": getattr(f, "filename", None)})
                continue

            # Reuse your existing single-image enrollment path
            res = PIPE.enroll_image(img, person_id=person_id, source=(getattr(f, "filename", None) or f"frame_{i:04d}.jpg"))
            total_faces += int(res.get("faces", 0))
            total_emb  += int(res.get("embeddings_added", 0))
            processed  += 1

        except Exception as e:
            errors += 1
            log.exception("register_frame_error", extra={"idx": i, "upload_filename": getattr(f, "filename", None), "error": str(e)})

    summary = {
        "ok": True,
        "person_id": person_id,
        "person_name": name,
        "received_files": len(files),
        "processed_files": processed,
        "errors": errors,
        "faces_detected": total_faces,
        "embeddings_added": total_emb,
        "capped_at": cap,
    }

    log.info("api_register_out", extra=summary)
    return summary

@app.post("/capture/video")
async def capture_video(
    person_id: str = Query(..., description="Employee/Person ID"),
    video: UploadFile = File(...),
    sample_every: int = Query(5, ge=1, le=60),
    max_frames: int = Query(150, ge=1, le=2000),
    min_face_px: int = Query(120, ge=60, le=1000),
):
    """
    Accept a video, extract frames at a constant rate, DETECT+ALIGN faces,
    save aligned crops, and return pose coverage + next UI instruction.
    """
    try:
        raw = video.file.read()
        if not raw:
            return JSONResponse(status_code=400, content={"error": "Empty video upload"})
        log.info(
            "api_capture_in",
            extra={
                "person_id": person_id,
                "upload_filename": getattr(video, "filename", None),
                "size_bytes": len(raw),
                "content_type": getattr(video, "content_type", None),
                "sample_every": sample_every,
                "max_frames": max_frames,
                "min_face_px": min_face_px,
            },
        )
        from app.core.capture import save_aligned_frames_from_video
        res = save_aligned_frames_from_video(
            raw,
            person_id=person_id,
            sample_every=sample_every,
            max_frames=max_frames,
            min_face_px=min_face_px,
        )
        log.info("api_capture_out", extra={"person_id": person_id, **res})
        return res
    except Exception as e:
        log.exception(
            "capture_video_failed",
            extra={"file": getattr(video, "filename", None), "error": str(e)},
        )
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    from app.clients.http import post_json
    from app import config

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

        # Decode image
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Invalid image upload {getattr(file, 'filename', None)}")

        # Recognize faces
        preds = PIPE.recognize_image(img)
        log.info("api_rec_out", extra={"faces": len(preds)})

        # --- Mark attendance for recognized person ---
        if preds and preds[0]["prediction"]["person_id"] != "unknown":
            person_id = preds[0]["prediction"]["person_id"]
            log.info("attendance_attempt", extra={"person_id": person_id})

            # Try to parse numeric ID, fallback to ERP_USER_MAP if defined
            try:
                user_id = int(person_id)
            except ValueError:
                user_id = getattr(config, "ERP_USER_MAP", {}).get(person_id)

            if user_id is None:
                msg = f"No ERP mapping for person_id '{person_id}'"
                log.warning("attendance_skipped", extra={"reason": msg})
                return {"faces": preds, "attendance": msg}

            # Send attendance request
            payload = {"user_id": user_id}
            log.info("attendance_request", extra={"payload": payload})
            status, resp = await post_json(config.ERP_ATTENDANCE_URL, payload)
            log.info("attendance_response", extra={"status": status, "resp": resp})

            return {"faces": preds, "attendance": resp}

        return {"faces": preds, "attendance": "no known face detected"}

    except Exception as e:
        log.exception("recognize_failed", extra={"file": getattr(file, "filename", None)})
        return JSONResponse(status_code=400, content={"error": str(e)})



# (Optional) Friendly root
@app.get("/")
def root():
    return {"message": "Attendance Face API", "try": ["/healthz", "/docs", "POST /enroll", "POST /enroll/video", "POST /recognize"]}
