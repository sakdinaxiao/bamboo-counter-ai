import cv2
from pathlib import Path

project_root = Path(__file__).resolve().parent

def apply_clahe(frame):
    clahe = cv2.createCLAHE(clipLimit=2.5,tileGridSize=(16,16))

    lab_img = cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)

    l,a,b = cv2.split(lab_img)
    l_enhanced = clahe.apply(l)

    merge = cv2.merge((l_enhanced,a,b))
    enhanced_img = cv2.cvtColor(merge,cv2.COLOR_LAB2BGR)


    return enhanced_img