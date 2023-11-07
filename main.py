from utils2 import *
import pandas as pd
import cv2

from icecream import ic

ROOT_DIR = "S:\GitHub\Visual-Odometry\saved_imgs"
FILTER_MATCHES = None
PLOT_TRAJECTORY = True

def main():
    imgDataset = datasetHandler(ROOT_DIR)

    failed_matches = 0
    matches = []
    keypoints = []
    descriptors = []

    for frameNumber in range(imgDataset.numImages):
        kp, des = extractFeatures(imgDataset.grayImages[frameNumber])
        keypoints.append(kp)
        descriptors.append(des)

    for frameNumber in range(imgDataset.numImages-1):
        print(f"Between Frame {frameNumber+1} and {frameNumber+2}:")
        match = matchFeatures(
            datasetHandler=imgDataset,
            frameNumber=frameNumber, 
            filter=FILTER_MATCHES,
        )
        if match in errorFlags:
            print(f"No matches found between frames {frameNumber+1} and {frameNumber+2}")
            failed_matches = failed_matches + 1
            matches.append("No Match")
        else:
            matches.append(match)

    print(f"Failed to match {failed_matches}/{len(matches)} frame pairs.")
    badIndices = [i for i, x in enumerate(matches) if x in errorFlags]

    trajectory = np.zeros((3, imgDataset.numImages))
    trajectory[:, 0] = np.array([0, 0, 0])
    for i in range(1, imgDataset.numImages):
        if i-1 or i in badIndices:
            trajectory[:, i] = trajectory[:, i-1]
        else:
            Rmat, tvec = estimateMotion(matches, keypoints, descriptors, i-1)
            trajectory[:, i] = trajectory[:, i-1] + np.dot(Rmat, tvec).reshape(-1)

    ic(trajectory.shape)

    if PLOT_TRAJECTORY:
        # Plot the trajectory in 3D
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(trajectory[0, :], trajectory[1, :], trajectory[2, :], 'b')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.text(trajectory[0, 0], trajectory[1, 0], trajectory[2, 0], "Path Start")
        ax.text(trajectory[0, -1], trajectory[1, -1], trajectory[2, -1], "Path End")
        ax.set_title('Camera Trajectory')
        plt.show()

if __name__ == "__main__":
    main()

