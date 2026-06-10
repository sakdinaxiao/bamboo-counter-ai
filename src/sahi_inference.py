from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
import supervision as sv
import torch


def get_available_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def get_sahi(model):
    return AutoDetectionModel.from_pretrained(
        model_type="yolov11",
        model_path= str(model),
        confidence_threshold=0.5,
        device=get_available_device()
    )

def apply_sahi(model,img):
    result = get_sliced_prediction(
        img,
        detection_model=model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.4,
        overlap_width_ratio=0.4,
        postprocess_type="NMS",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=0.4,
        verbose=False
    )

    xyxy = []
    confidence = []
    class_id=[]

    for i in result.object_prediction_list:
        xyxy.append([i.bbox.minx,i.bbox.miny, i.bbox.maxx, i.bbox.maxy])
        confidence.append(i.score.value)
        class_id.append(i.category.id)

    if len(xyxy) > 0 :
        return sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidence),
            class_id=np.array(class_id)
        )
    else:
        return sv.Detections.empty()
