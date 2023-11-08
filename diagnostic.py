from utils import *
from icecream import ic

rootDir = "S:\GitHub\Visual-Odometry\data"
dataset = dataManager(rootDir)

trajectory = dataset.plotTrajectory()

ic(trajectory.shape)
ic(trajectory)



