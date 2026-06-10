import cv2
import numpy as np
from scipy.spatial import cKDTree


cv2.setRNGSeed(42)

class GlobalTracker:
    def __init__(self,merge_distance=8.0):
        self.merge_distance = merge_distance
        self.map = []
        self.prev_frame = None
        self.total_camera_movement = np.identity(3)

        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)

    def find_camera_movement(self,frame):
        grey_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        if self.prev_frame is None:
            self.prev_frame = grey_frame
            return np.array([[1,0,0],
                             [0,1,0]])
        
        else:
            keypoint_old , descriptor_old = self.feature_detector.detectAndCompute(self.prev_frame,None)
            keypoint_new , descriptor_new = self.feature_detector.detectAndCompute(grey_frame,None)

        if len(keypoint_old) < 10 or len(keypoint_new) <10 or descriptor_new is None or descriptor_old is None :
            print("not enough point")
            self.prev_frame = grey_frame
            return np.array([[1,0,0],
                             [0,1,0]])
            
        matches = self.feature_matcher.knnMatch(descriptor_old,descriptor_new, k=2)
        
        good_match = []
        for pairs in matches:
            if len(pairs) == 2:
                first , second = pairs
                if first.distance < 0.75 * second.distance:
                    good_match.append(first)
        
        
        old_point = np.float32([keypoint_old[m.queryIdx].pt for m in good_match]).reshape(-1,1,2)
        new_point = np.float32([keypoint_new[m.trainIdx].pt for m in good_match]).reshape(-1,1,2)

        tranform_matrix , _ = cv2.estimateAffinePartial2D(
            old_point,
            new_point,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=200,
            confidence=0.99
        )

        if tranform_matrix is None:
            print("couldn't calculate movement")
            self.prev_frame = grey_frame
            return np.array([[1,0,0],
                             [0,1,0]])
        
        self.prev_frame = grey_frame
        return tranform_matrix #3x3
    
    def update_global_movement(self,frame_movement):
        movement33 = np.identity(3, dtype=np.float32)
        movement33[0:2, :] = frame_movement
        invmovement = np.linalg.inv(movement33)

        self.total_camera_movement = self.total_camera_movement @ invmovement

    def update(self,frame,detection):
        current_tranform = self.find_camera_movement(frame=frame)
        self.update_global_movement(current_tranform)

        if detection is None or len(detection.xyxy) == 0 :
            return self.map

        new_center = []
        for box in detection.xyxy:
            cx = (box[0]+box[2])/2.0
            cy = (box[1] + box[3])/2.0

            new_center.append([cx,cy,1.0])

        global_center = self.total_camera_movement @ np.array(new_center).T
        global_center = global_center[0:2, :].T

        update_rate = 0.25
        tree = cKDTree(self.map) if len(self.map) > 0 else None

        for p in global_center:
            px, py = float(p[0]), float(p[1])

            if tree is None:
                self.map.append([px, py])
                continue

            dist, idx = tree.query([px, py])
            if dist <= self.merge_distance:
                # nudge the anchor toward the new observation so it follows
                
                ox, oy = self.map[idx]
                self.map[idx] = [
                    (1 - update_rate) * ox + update_rate * px,
                    (1 - update_rate) * oy + update_rate * py,
                ]
            else:
                self.map.append([px, py])

        return self.map
    
    def count(self):
        return len(self.map)