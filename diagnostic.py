from utils import *
from icecream import ic

rootDir = "N:\GitHub\Visual-Odometry\data"
dataset = dataManager(rootDir)

trajectory = np.zeros((3, dataset.numImages))
for i in range(dataset.numImages-1):
    q1, q2 = dataset.getMatches(i)
    transf = dataset.getPose(q1, q2)
    trajectory[:, i] = transf[:3, 3]

# Plot the trajectory
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(trajectory[0], trajectory[1], trajectory[2], label='Trajectory')
ax.legend()
plt.show()

