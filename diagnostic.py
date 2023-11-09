from utils import *
from icecream import ic


ROOT_DIR = "S:\\GitHub\\Visual-Odometry"
dataset = dataManager(ROOT_DIR)

runNumber = 2

trajectory = dataset.readTrajectory(runNumber)
# Plot Trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f"Trajectory")
ax.legend()
ax.text( trajectory[0, 0],  trajectory[1, 0],  trajectory[2, 0], "Start")
ax.text(trajectory[0, -1], trajectory[1, -1], trajectory[2, -1],   "End")
plt.show()
