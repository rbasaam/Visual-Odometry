from utils import *

ROOT_DIR = "N:\\GitHub\\Visual-Odometry"
FEATURE_DETECTOR = "ORB" # "ORB" or "SIFT"

ANIMATE_FRAMES = False
ANIMATION_FPS = 30.0
ANIMATION_FILE = None

CREATE_LOG = True
SAVE_IMG = True

def main():
    
    dataset = dataManager(
        rootDir = ROOT_DIR,
        featureDetector=FEATURE_DETECTOR
        )

    if ANIMATE_FRAMES:
        dataset.animateFrames()
    
    visualPath = dataset.trackMotion(
        plottingFlag=SAVE_IMG,
        loggingFlag=CREATE_LOG
    )

    return

if __name__ == "__main__":
    main()

