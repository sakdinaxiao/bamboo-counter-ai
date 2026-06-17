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
