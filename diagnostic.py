from utils import *
from icecream import ic

ROOT_DIR = "S:\\GitHub\\Visual-Odometry"
FEATURE_DETECTOR = "ORB" # "ORB" or "SIFT"

dataset = dataManager(rootDir = ROOT_DIR)
dataset._loadImages()
detector, index_params, search_params = dataset._createFeatureDetector(FEATURE_DETECTOR)
kp, des = detector.detectAndCompute(dataset.grayImages[0], None)

# Draw keypoints
img = cv2.drawKeypoints(dataset.grayImages[0], kp, None, color=(0,255,0), flags=0)
cv2.imshow("Keypoints", img)
# Save Image
cv2.imwrite("Keypoints.png", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

