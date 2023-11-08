from utils import *
from icecream import ic

rootDir = "S:\GitHub\Visual-Odometry\data"
dataset = dataManager(rootDir)
# dataset.sampleFrame(1)
trajectory = dataset.plotTrajectory(featureDetector="SIFT")

ic(trajectory.shape)
ic(trajectory)
