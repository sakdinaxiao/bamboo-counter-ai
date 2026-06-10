# Bamboo AI

computer vision project for detecting, tracking, and counting bamboo in video.

This repository is still a work in progress. The current goal is to test a pipeline that combines:

- YOLO-based segmentation
- SAHI sliced detection
- Camera-motion-compensated counting via GlobalTracker
- CLAHE image enhancement


## Current Status

This project is not a polished package yet. It is closer to an experimental repo for model training and inference testing.



## Project Files

- [main.py](/Users/jacky/bamboo_ai/main.py) runs the main inference and counting pipeline on a video
- [segmentation.py](/Users/jacky/bamboo_ai/segmentation.py) extracts segmentation regions used before detection
- [sahi_inference.py](/Users/jacky/bamboo_ai/sahi_inference.py) runs sliced object detection with SAHI
- [global_tracker.py](/Users/jacky/bamboo_ai/global_tracker.py) accumulates detections in a global coordinate map and counts unique bamboo
- [clahe_inference.py](/Users/jacky/bamboo_ai/clahe_inference.py) applies CLAHE image enhancement
- [train_detect.py](/Users/jacky/bamboo_ai/train_detect.py) trains and evaluates the detection model


## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running Inference

1. Put your trained weights in the expected folders:

- `training_result/detection_small/weights/best.pt`
- `training_result/segment/best.pt`

2. Put your input video inside the project folder.

3. Run:

```bash
python main.py --source your_video_name.mp4
```

Example:

```bash
python main.py --source water.MP4
```

The `--source` value should be the video filename or relative path from the project root.



## How counting works

Rather than tracking objects frame-to-frame with IDs, the pipeline uses a global spatial map:

1. **Camera registration** — ORB features are matched between consecutive frames and `estimateAffinePartial2D` (RANSAC) estimates the camera motion.
2. **Global coordinate mapping** — each detection center is projected into a global coordinate space that cancels out camera movement, so bamboo that stays stationary in the real world maps to the same point across frames.
3. **Spatial deduplication** — a cKDTree checks every new detection against existing map points. Points within `merge_distance` pixels update the existing anchor (centroid averaging); points farther away are added as new entries.
4. **Count** — `len(map)` at the end is the total number of unique bamboo seen across the whole video.

This approach works well for overhead drone video with a slowly panning camera and densely packed bamboo, where ID-based trackers struggle with occlusion and scale.


## Notes

- Device selection falls back between `mps`, `cuda`, and `cpu`
- Output video is saved as `<input_stem>_tracked_1080p.mp4` in the project root


## Limitations

- this is still an in-progress research/project repo
- model weights are expected locally and are not packaged cleanly yet
- some scripts are still tailored to the current dataset layout
- there are not yet automated tests

