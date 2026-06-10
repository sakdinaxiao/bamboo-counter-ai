from ultralytics import YOLO
import numpy as np
import cv2

def segmenting(model,frame):
    PADDING = 50

    seg_results = model(frame, verbose=False,conf=0.25)[0]
    if seg_results.masks is None:
        return []

    h_img,w_img = frame.shape[:2]
    mask = np.zeros((h_img,w_img), dtype=np.uint8)

    for m in seg_results.masks.data:
            mask = np.maximum(mask, cv2.resize(m.cpu().numpy(), (w_img, h_img)))

    contours, _ = cv2.findContours((mask > 0.25).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 1 or h < 1: continue

        x_min, y_min = max(0, x - PADDING), max(0, y - PADDING)
        x_max, y_max = min(w_img, x + w + PADDING), min(h_img, y + h + PADDING)

        cropped_roi = frame[y_min:y_max, x_min:x_max]

        regions.append({
                "image": cropped_roi,
                "offset_x": x_min,
                "offset_y": y_min
            })
        
    return regions
