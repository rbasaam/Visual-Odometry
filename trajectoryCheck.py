import os
import numpy as np
import pandas as pd
from icecream import ic
import matplotlib.pyplot as plt

trajectoryLogFile = "C:\\Users\\rbasa\\Documents\\GitHub\\Visual-Odometry\\logs\\results\\ORB_trajectory_1.txt"

trajectoryLog = np.array(pd.read_csv(trajectoryLogFile, sep=",", header=None))
cumulativeSum = np.cumsum(trajectoryLog, axis=0)
ic(cumulativeSum)
ic(cumulativeSum.shape)

# Plot trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(cumulativeSum[:, 0], cumulativeSum[:, 1], cumulativeSum[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
