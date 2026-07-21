import supervision as sv

def get_bytetrack():
    tracker =  sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.5,
            minimum_consecutive_frames=2
)
    counter = set()


    return tracker, counter

def get_counting_zone(frame_shape, margin_ratio=0.05):
    height, width = frame_shape[:2]
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio

    return (
        margin_x,
        margin_y,
        width - margin_x,
        height - margin_y,
    )

def tracking(detected,bytetrackmodel,counter,counting_zone):
    track =  bytetrackmodel.update_with_detections(detected)
    min_x, min_y, max_x, max_y = counting_zone

    if track.tracker_id is not None:
        for bbox, tracker_id in zip(track.xyxy, track.tracker_id):
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            if (min_x < center_x < max_x) and (min_y < center_y < max_y):
                counter.add(tracker_id)

    
    return track
