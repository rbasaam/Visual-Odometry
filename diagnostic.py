from utils import *
from icecream import ic

rootDir = "N:\\GitHub\\Visual-Odometry\\data"
dataset = dataManager(dataDir=rootDir, featureDetector="ORB")
# dataset.animateFrames()
dataset.trackMotion()
