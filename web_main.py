import shutil
import subprocess
from pathlib import Path
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from src.sahi_inference import apply_sahi, get_sahi
from src.clahe_inference import apply_clahe
from src.global_tracker import GlobalTracker
from src.segmentation import segmenting

project_root = Path(__file__).resolve().parent
_model_path = project_root / "training_result" / "detection_small" / "weights" / "best.pt"
_seg_model_path = project_root / "training_result" / "segment" / "best.pt"

_ffmpeg = shutil.which("ffmpeg")


def _to_h264(src: str, dst: str) -> None:
    # cv2 only writes mp4v, which browsers cannot decode; transcode to H.264
    if _ffmpeg:
        subprocess.run(
            [_ffmpeg, "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-an", dst],
            check=True, capture_output=True,
        )
        Path(src).unlink(missing_ok=True)
    else:
        Path(src).replace(dst)


def run(video_path: str, output_path: str | None = None) -> int:
    if not _model_path.exists():
        raise FileNotFoundError(f"Detection model not found: {_model_path}")

    use_segmentation = _seg_model_path.exists()
    seg_model = YOLO(str(_seg_model_path)) if use_segmentation else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    writer = None
    box_annotator = None
    raw_path = None

    try:
        if output_path:
            raw_path = str(Path(output_path).with_name("raw_" + Path(output_path).name))
            origin_fps_pre = cap.get(cv2.CAP_PROP_FPS) or 30
            stride_pre = max(1, int(origin_fps_pre / 3))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_fps = max(1.0, origin_fps_pre / stride_pre)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(raw_path, fourcc, out_fps, (w, h))
            box_annotator = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=2)

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

            if writer is not None:
                annotated = frame.copy()
                if not detections.is_empty():
                    annotated = box_annotator.annotate(annotated, detections)
                cv2.putText(
                    annotated, f"Count: {tracker.count()}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2,
                    cv2.LINE_AA,
                )
                writer.write(annotated)

        if writer is not None:
            writer.release()
            writer = None
            _to_h264(raw_path, output_path)

        return tracker.count()
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if raw_path is not None:
            Path(raw_path).unlink(missing_ok=True)
