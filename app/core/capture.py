# app/core/capture.py
from __future__ import annotations
import time, uuid, os, cv2
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

from app.logging_config import get_logger, timed
from app.core.video import iter_video_frames_from_bytes
from app.core.detector import FaceDetector
from app.core.align import align_5pts
from app import config

log = get_logger(__name__)

def _classify_pose_heuristic(kps: np.ndarray) -> str:
    """
    Very lightweight pose bucket from 5 keypoints (left_eye, right_eye, nose, left_mouth, right_mouth).
    Heuristics only, just to track coverage for instructions.
    """
    le, re, nose, lm, rm = kps.astype(np.float32)
    eye_mid = (le + re) / 2.0
    mouth_mid = (lm + rm) / 2.0
    face_h = np.linalg.norm(mouth_mid - eye_mid) + 1e-6
    dx = (nose[0] - eye_mid[0]) / face_h
    dy = (nose[1] - eye_mid[1]) / face_h
    # thresholds tuned for rough guidance
    if dx > 0.18:   return "right"
    if dx < -0.18:  return "left"
    if dy < -0.12:  return "up"
    if dy > 0.18:   return "down"
    return "center"

def save_aligned_frames_from_video(
    video_bytes: bytes,
    person_id: str,
    sample_every: int = 5,
    max_frames: int = 150,
    jpeg_quality: int = 95,
    min_face_px: int = 120,
) -> Dict[str, Any]:
    """
    Extract frames at a constant rate, DETECT the largest face, ALIGN to 112x112,
    and save aligned crops to:
      data/faces/<person_id>/captures/<session_id>/aligned_XXXX_<pose>.jpg
    Returns counts and pose coverage for UI feedback.
    """
    # session directory
    session_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    root = config.FACES_DIR / person_id / "captures" / session_id
    root.mkdir(parents=True, exist_ok=True)

    # decode + sample
    frames = iter_video_frames_from_bytes(
        video_bytes, sample_every=sample_every, max_frames=max_frames
    )

    det = FaceDetector()
    kept = 0
    pose_counts = {"left":0, "right":0, "up":0, "down":0, "center":0}
    saved_paths: List[str] = []
    imwrite_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    with timed(log, "capture_align_save", person_id=person_id, session=str(root)):
        for idx, fr in enumerate(frames, start=1):
            bboxes, kps = det.detect(fr)
            if bboxes.shape[0] == 0:
                continue
            # pick biggest
            areas = (bboxes[:,2]-bboxes[:,0]) * (bboxes[:,3]-bboxes[:,1])
            j = int(np.argmax(areas))
            w = bboxes[j][2]-bboxes[j][0]
            h = bboxes[j][3]-bboxes[j][1]
            if min(w, h) < min_face_px:
                continue

            pose = _classify_pose_heuristic(kps[j]) if kps.shape[0] > 0 else "center"
            pose_counts[pose] = pose_counts.get(pose, 0) + 1

            # align → 112x112
            aligned = align_5pts(fr, kps[j])
            if aligned is None:
                continue
            if aligned.shape[:2] != (config.ALIGNED_SIZE, config.ALIGNED_SIZE):
                aligned = cv2.resize(aligned, (config.ALIGNED_SIZE, config.ALIGNED_SIZE))

            out = root / f"aligned_{idx:04d}_{pose}.jpg"
            cv2.imwrite(str(out), aligned, imwrite_params)
            saved_paths.append(str(out))
            kept += 1

    # Suggest next instruction based on what’s missing
    order = ["center", "left", "right", "up", "down"]
    want_per_bucket = 3  # tweak if you want more per pose
    next_instr = None
    for bucket in order:
        if pose_counts.get(bucket, 0) < want_per_bucket:
            next_instr = f"look {bucket}"
            break

    log.info(
        "capture_aligned_done",
        extra={
            "person_id": person_id,
            "session": str(root),
            "frames_total": len(frames),
            "aligned_saved": kept,
            **{f"pose_{k}": v for k,v in pose_counts.items()},
            "next_instruction": next_instr or "done",
        },
    )
    return {
        "person_id": person_id,
        "session_dir": str(root),
        "frames_total": len(frames),
        "aligned_saved": kept,
        "pose_counts": pose_counts,
        "next_instruction": next_instr or "done",
        "sample_every": sample_every,
        "max_frames": max_frames,
        "min_face_px": min_face_px,
        "jpeg_quality": jpeg_quality,
        "saved_paths": saved_paths,  # handy for debugging; remove later if noisy
    }
