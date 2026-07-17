from src.clahe_inference import apply_clahe
from ultralytics import YOLO
from src.sahi_inference import SahiDetector
from pathlib import Path
import cv2
import supervision as sv
# from bytetrack_inference import get_bytetrack, get_counting_zone, tracking
from src.segmentation import segmenting
import numpy as np
import argparse
from tracker import Tracker

project_root = Path(__file__).resolve().parent.parent
TARGET_FPS = 3

#using small YOLO
def main(args):
    model_path = project_root / args.model_path
    seg_model_path = project_root / args.seg_model_path

    if not seg_model_path.exists():
        print("Segmentation model path does not exist")
        return
    seg_model = YOLO(seg_model_path)
    if not model_path.exists():
        print("Model path does not exist")
        return


    source_vid = project_root / args.source
    if not source_vid.exists():
        print("There no video")
        return

    cap = cv2.VideoCapture(str(source_vid))
    if not cap.isOpened():
        print("Can't open video")
        return

    output_path = project_root / f"{source_vid.stem}_tracked_1080p.mp4"
    video_writer = None
    
    try:

        sahi_detector = SahiDetector(
            model_path=model_path,
            confidence_threshold=args.sahi_conf,
            slice_height=args.sahi_slice,
            slice_width=args.sahi_slice,
            overlap_height_ratio=args.sahi_overlap,
            overlap_width_ratio=args.sahi_overlap,
            postprocess_match_threshold=args.sahi_match_thresh
        )

        tracker = Tracker(
            merge_distance=args.merge_distance,
            update_rate=args.update_rate,
            ransac_threshold=args.ransac_threshold,
            orb_features=args.orb_features
        )
        
        #----- for visual only
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        #-----

        origin_fps = cap.get(cv2.CAP_PROP_FPS)
        stride = max(1, int(origin_fps/TARGET_FPS))
        output_size = (1920, 1080)
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_path), fourcc, TARGET_FPS, output_size)

        if not video_writer.isOpened():
            print("Can't create output video")
            return

        print(f"Saving output video to: {output_path}")

        frame_count = 0

        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                break
            
            frame_count += 1

            if frame_count % stride != 0:
                continue


            regions = segmenting(seg_model,frame)
            frame = apply_clahe(frame=frame)
            all_detections_xyxy = []
            all_confidence = []
            all_class_ids=[]

            for region in regions:
                reg_img = region["image"]
                offset_x = region["offset_x"]
                offset_y = region["offset_y"]

                reg_img = apply_clahe(reg_img)
                detected = sahi_detector.predict(reg_img)

                if not detected.is_empty():
                    for i in range(len(detected.xyxy)):
                        remaped=[
                            detected.xyxy[i][0] + offset_x,
                            detected.xyxy[i][1] + offset_y,
                            detected.xyxy[i][2] + offset_x,
                            detected.xyxy[i][3] + offset_y 
                        ]
                        all_detections_xyxy.append(remaped)
                        all_confidence.append(detected.confidence[i])
                        all_class_ids.append(detected.class_id[i])

            if len(all_detections_xyxy) > 0:
                final_detections = sv.Detections(
                    xyxy=np.array(all_detections_xyxy),
                    confidence=np.array(all_confidence),
                    class_id=np.array(all_class_ids)
                )
            else:
                final_detections = sv.Detections.empty()

            print(f"Raw YOLO Detections this frame: {len(all_detections_xyxy)}")

            tracker.update(frame,final_detections)


            # -------------------------- Visulization

            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=final_detections)
            cv2.putText(
                annotated_frame,
                f"Total Counted: {tracker.count()}",
                (40, 70), 
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0, 
                (0, 255, 0), 
                4 
            )
            
            cv2.imshow("Tracking & Counting Debugger", annotated_frame)
            output_frame = cv2.resize(annotated_frame, output_size, interpolation=cv2.INTER_LINEAR)
            video_writer.write(output_frame)

            if cv2.waitKey(1) == ord('q'):
                break
            
            # ---------------------------
        
        return tracker.count()
    
    except Exception as e:
        print(f"Error {e}")
        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bamboo counter")
    parser.add_argument("--source", type=str, required=True, help="Path to source video")
    
    # Model paths
    parser.add_argument("--model_path", type=str, default="training_result/detection_small/weights/best.pt", help="Path to detection model")
    parser.add_argument("--seg_model_path", type=str, default="training_result/segment/best.pt", help="Path to segmentation model")
    
    # Tracker parameters
    parser.add_argument("--merge_distance", type=float, default=12.5)
    parser.add_argument("--update_rate", type=float, default=0.25)
    parser.add_argument("--ransac_threshold", type=float, default=5.0)
    parser.add_argument("--orb_features", type=int, default=1000)
    
    # SAHI parameters
    parser.add_argument("--sahi_conf", type=float, default=0.6)
    parser.add_argument("--sahi_slice", type=int, default=640)
    parser.add_argument("--sahi_overlap", type=float, default=0.4)
    parser.add_argument("--sahi_match_thresh", type=float, default=0.4)
    
    args = parser.parse_args()
    print(f"This video have {main(args)} bamboos")
    
