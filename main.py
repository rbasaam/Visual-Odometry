from utils import *

ROOT_DIR = "S:\\GitHub\\Visual-Odometry"

FEATURE_DETECTOR = "ORB" # "ORB" or "SIFT"

ANIMATE_FRAMES = False
ANIMATION_FPS = 30.0

SAVE_MATCHES = True

def main():
    
    dataset = dataManager(
        rootDir = ROOT_DIR,
        )

    if ANIMATE_FRAMES:
        dataset.animateFrames()
    
    visualPath = dataset._trackMotion(
        featureDetector = FEATURE_DETECTOR,
        saveMatches = SAVE_MATCHES,
        )

    return

if __name__ == "__main__":
    main()

