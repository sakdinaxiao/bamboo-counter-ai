import cv2
import numpy as np
from scipy.spatial import cKDTree
import logging

logger = logging.getLogger(__name__)


class GlobalSpatialMapper:
    """
    A class to track camera movement and maintain a global map of detected objects.

    Attributes:
        merge_distance (float): Distance threshold for merging nearby points.
        update_rate (float): Rate at which existing points are updated towards new observations.
        ransac_threshold (float): RANSAC reprojection threshold for affine transform estimation.
        orb_features (int): Number of ORB features to retain.
        map (list): A list of globally tracked points representing object locations.
        prev_frame (numpy.ndarray): The previous grayscale frame for feature matching.
        total_camera_movement (numpy.ndarray): Cumulative 3x3 transformation matrix of the camera.
        prev_transform (numpy.ndarray): The last successfully computed 2x3 transformation matrix.
        feature_detector (cv2.ORB): The ORB feature detector.
        feature_matcher (cv2.BFMatcher): The brute-force matcher for feature matching.
    """

    def __init__(self, merge_distance=8.0, update_rate=0.25, ransac_threshold=5.0, orb_features=1000):
        """
        Initializes the GlobalSpatialMapper with given parameters.

        Args:
            merge_distance (float): Distance threshold to consider two points as the same object.
            update_rate (float): Weight for updating existing point coordinates with new observations.
            ransac_threshold (float): Threshold for RANSAC in affine transform estimation.
            orb_features (int): Maximum number of ORB features to detect.
        """
        self.merge_distance = merge_distance
        self.update_rate = update_rate
        self.ransac_threshold = ransac_threshold
        self.orb_features = orb_features
        self.map = []
        self.prev_frame = None
        self.total_camera_movement = np.identity(3)
        self.prev_transform = np.array([[1.0, 0.0, 0.0], 
                                        [0.0, 1.0, 0.0]])

        self.feature_detector = cv2.ORB_create(nfeatures=self.orb_features)
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def find_camera_movement(self, frame):
        """
        Estimates the camera movement between the previous frame and the current frame.

        Args:
            frame (numpy.ndarray): The current BGR image frame.

        Returns:
            numpy.ndarray: A 2x3 affine transformation matrix representing the camera movement.
        """
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame is None:
            self.prev_frame = grey_frame
            return self.prev_transform
        
        else:
            keypoint_old, descriptor_old = self.feature_detector.detectAndCompute(self.prev_frame, None)
            keypoint_new, descriptor_new = self.feature_detector.detectAndCompute(grey_frame, None)

        if len(keypoint_old) < 10 or len(keypoint_new) < 10 or descriptor_new is None or descriptor_old is None:
            logger.warning("not enough point")
            self.prev_frame = grey_frame
            return self.prev_transform
            
        matches = self.feature_matcher.knnMatch(descriptor_old, descriptor_new, k=2)
        
        good_match = []
        for pairs in matches:
            if len(pairs) == 2:
                first, second = pairs
                if first.distance < 0.75 * second.distance:
                    good_match.append(first)

        old_point = np.float32([keypoint_old[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
        new_point = np.float32([keypoint_new[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)

        tranform_matrix, _ = cv2.estimateAffinePartial2D(
            old_point,
            new_point,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            maxIters=200,
            confidence=0.99
        )

        if tranform_matrix is None:
            logger.warning("couldn't calculate movement")
            self.prev_frame = grey_frame
            return self.prev_transform
        
        self.prev_frame = grey_frame
        self.prev_transform = tranform_matrix
        return tranform_matrix
    
    def update_global_movement(self, frame_movement):
        """
        Updates the total cumulative camera movement matrix using the frame-to-frame movement.

        Args:
            frame_movement (numpy.ndarray): The 2x3 affine transformation matrix from the latest frame.
        """
        movement33 = np.identity(3, dtype=np.float32)
        movement33[0:2, :] = frame_movement
        invmovement = np.linalg.inv(movement33)

        self.total_camera_movement = self.total_camera_movement @ invmovement

    def update(self, frame, detection):
        """
        Updates the global map of objects based on camera movement and new detections.

        Args:
            frame (numpy.ndarray): The current BGR image frame.
            detection: An object containing detection bounding boxes in its `xyxy` attribute.

        Returns:
            list: The updated map of globally tracked object points.
        """
        current_tranform = self.find_camera_movement(frame=frame)
        self.update_global_movement(current_tranform)

        if detection is None or len(detection.xyxy) == 0:
            return self.map

        new_center = []
        for box in detection.xyxy:
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0

            new_center.append([cx, cy, 1.0])

        global_center = self.total_camera_movement @ np.array(new_center).T
        global_center = global_center[0:2, :].T

        update_rate = self.update_rate
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
        """
        Gets the total count of distinct objects tracked in the global map.

        Returns:
            int: The number of tracked objects.
        """
        return len(self.map)