from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
import supervision as sv
import torch

class SahiDetector:
    def __init__(self, 
                 model_path, 
                 confidence_threshold=0.6, 
                 slice_height=640, 
                 slice_width=640, 
                 overlap_height_ratio=0.4, 
                 overlap_width_ratio=0.4, 
                 postprocess_match_threshold=0.4):
                 
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.postprocess_match_threshold = postprocess_match_threshold
        
        device = self._get_available_device()
        
        self.model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(model_path),
            confidence_threshold=confidence_threshold,
            device=device
        )
        
    def _get_available_device(self):
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
        
    def predict(self, img):
        result = get_sliced_prediction(
            img,
            detection_model=self.model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=self.postprocess_match_threshold,
            verbose=False
        )

        xyxy = []
        confidence = []
        class_id = []

        for i in result.object_prediction_list:
            xyxy.append([i.bbox.minx, i.bbox.miny, i.bbox.maxx, i.bbox.maxy])
            confidence.append(i.score.value)
            class_id.append(i.category.id)

        if len(xyxy) > 0:
            return sv.Detections(
                xyxy=np.array(xyxy),
                confidence=np.array(confidence),
                class_id=np.array(class_id)
            )
        else:
            return sv.Detections.empty()
