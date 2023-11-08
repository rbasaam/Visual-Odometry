from utils import *
import pandas as pd
import cv2

from icecream import ic

ROOT_DIR = "S:\GitHub\Visual-Odometry\data"
FEATURE_DETECTOR = "ORB" # "ORB" or "SIFT"

def main():
    
    dataset = dataManager(ROOT_DIR)
    dataset.sampleFrame(1)
    trajectory = dataset.plotTrajectory(featureDetector=FEATURE_DETECTOR)

    ic(trajectory.shape)
    ic(trajectory)

    return

if __name__ == "__main__":
    main()

