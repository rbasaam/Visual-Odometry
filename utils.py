import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

errorMsgs = {
    "err_DescriptorLength": "Descriptor length is less than 2, the size of the kd-tree cannot be less than 2.\n",
    "err_NoneDescriptor": "One of the descriptors is None, cannot compute matches.\n",
    "err_InsufficientMatches": "Number of matches is less than 6, cannot compute Essential Matrix.\n"
}

class dataManager():
    def __init__(self, rootDir: str, featureDetector: str = "ORB"):
        # Define Relevant Directories
        self.dataDir = os.path.join(rootDir, "data")
        self.logDir = os.path.join(rootDir, "logs")
        self.resultsDir = os.path.join(self.logDir, "results")
        self.plotDir = os.path.join(self.logDir, "plots")
        self.runtimeDir = os.path.join(self.logDir, "runtime")
        # Get the List of Image Filenames to Read
        self.rgbList = [os.path.join(os.path.join(self.dataDir, "pov"), rgbFile) for rgbFile in os.listdir(os.path.join(self.dataDir, "pov"))]
        self.depthList = [os.path.join(os.path.join(self.dataDir, "depth"), depthFile) for depthFile in os.listdir(os.path.join(self.dataDir, "depth"))]

        # Get the Number of Frames in the Dataset
        self.numImages = len(self.rgbList)

        self.grayImages = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in self.rgbList]
        self.depthImages = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in self.depthList]

        self.featureDetector = featureDetector
    
    def sampleFrame(self, frameNum: int):
        
        rgbImage = cv2.cvtColor(cv2.imread(self.rgbList[frameNum-1]), cv2.COLOR_BGR2RGB)
        grayImage = cv2.imread(self.rgbList[frameNum-1], cv2.IMREAD_GRAYSCALE)
        depthImage = cv2.imread(self.depthList[frameNum-1], cv2.IMREAD_GRAYSCALE)

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(rgbImage)
        ax[0].set_title(f"Frame {frameNum} RGB") 
        ax[1].imshow(grayImage, cmap='gray')
        ax[1].set_title(f"Frame {frameNum} Grayscale")
        ax[2].imshow(depthImage, cmap='gray')
        ax[2].set_title(f"Frame {frameNum} Depth")
        plt.show()
        return
    
    def animateFrames(self, fps=20.0, outputFilename = None):
        if outputFilename is None:
            outputFilename = "frameAnimation.mp4"

        height, width = self.grayImages[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(outputFilename, fourcc, fps, (width, height))
        for f in range(self.numImages):
            frame =cv2.imread(self.rgbList[f], cv2.COLOR_BGR2RGB)
            video.write(frame)
        video.release()
        return

       
    def trackMotion(self, plottingFlag=True, loggingFlag=True):

        trajectory = np.zeros((3, self.numImages))

        if loggingFlag:
            # Create New Log Files
            logNumber = len(os.listdir(self.plotDir))+1

            logFileName = os.path.join(self.runtimeDir, f"{self.featureDetector}_log_{logNumber}.txt")
            resultsFilename = os.path.join(self.resultsDir, f"{self.featureDetector}_path_{logNumber}.txt")
            plotFilename = os.path.join(self.plotDir, f"{self.featureDetector}_trajectory_{logNumber}.png")

            logFile = open(logFileName, "w")
            resultsFile = open(resultsFilename, "w")

        if self.featureDetector == "ORB":
            # Initiate ORB detector
            detector = cv2.ORB_create(3000)
            # Match descriptors.
            flannIndex_LSH = 6
            index_params = dict(algorithm=flannIndex_LSH, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)

        elif self.featureDetector == "SIFT":
            # Initiate SIFT detector
            detector = cv2.SIFT_create(3000)
            # Match descriptors.
            index_params = dict(algorithm=0, trees=20)
            search_params = dict(checks=150)

        for frameNum in range(1, self.numImages):
            if loggingFlag:
                with open(logFileName, "a") as logFile:
                    logFile.write(f"------------------Frame {frameNum}------------------\n")

            # Find the keypoints and descriptors with Detector
            kp1, des1 = detector.detectAndCompute(self.grayImages[frameNum-1], None)
            kp2, des2 = detector.detectAndCompute(self.grayImages[frameNum], None)

            # Create FLANN matcher
            flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

            if des1 is None or des2 is None:
                matches = 'err_NoneDescriptor'
            elif len(des1) < 2 or len(des2) < 2:
                matches = 'err_DescriptorLength'
            else:
                matches = flann.knnMatch(des1, des2, k=2)
        
            if matches in errorMsgs.keys():
                if loggingFlag:
                    with open(logFileName, "a") as logFile:
                        logFile.write(errorMsgs[matches])
                
                q1 = np.array([])
                q2 = np.array([])

            else:
                # Apply ratio test
                goodMatches = []
                try:
                    for m, n in matches:
                        if m.distance < 0.7*n.distance:
                            goodMatches.append(m)
                except ValueError:
                    pass

                draw_params = dict(matchColor = -1, # draw matches in green color
                        singlePointColor = None,
                        matchesMask = None, # draw only inliers
                        flags = 2)

                img3 = cv2.drawMatches(self.grayImages[frameNum-1], kp1, self.grayImages[frameNum],kp2, goodMatches ,None,**draw_params)
                cv2.imshow("image", img3)
                cv2.waitKey(200)

                # Get the image points form the good matches
                q1 = np.float32([kp1[m.queryIdx].pt for m in goodMatches])
                q2 = np.float32([kp2[m.trainIdx].pt for m in goodMatches])
            
            numMatches = q1.shape[0]
            if numMatches < 6:
                trajectory[:, frameNum] = trajectory[:, frameNum-1]
                if loggingFlag:
                    with open(logFileName, "a") as logFile:
                        logFile.write(f"Number of Matches with Frame {frameNum+1}: {numMatches}\n")
                        logFile.write(errorMsgs["err_InsufficientMatches"])
            else:
                E, mask = cv2.findEssentialMat(q1, q2, method=cv2.RANSAC, prob=0.999, threshold=0.1)
                _, R, t, mask = cv2.recoverPose(E, q1, q2)
                trajectory[:, frameNum] = t.flatten()
                if loggingFlag:
                    with open(logFileName, "a") as logFile:
                        logFile.write(f"Number of Matches with Frame {frameNum+1}: {numMatches}\n")
                        logFile.write(f"Number of Inliers with Frame {frameNum+1}: {np.sum(mask)}\n")
                        logFile.write(f"Number of Good Matches with Frame {frameNum+1}: {len(goodMatches)}\n")
                        logFile.write(f"Essential Matrix Shape {E.shape}\n")
                        logFile.write(f"Essential Matrix \n {E} \n")
                        logFile.write(f"Rotation Matrix Shape {R.shape}\n")
                        logFile.write(f"Rotation Matrix \n {R} \n")
                        logFile.write(f"Translation Matrix Shape {t.shape}\n")
                        logFile.write(f"Translation Matrix \n {t}\n")

        path = np.cumsum(trajectory, axis=1)
        
        if loggingFlag:
            with open(resultsFilename, "a") as resultsFile:
                np.savetxt(resultsFile, path.transpose(), delimiter=",", newline="\n", fmt="%.4f")
                
        if plottingFlag:
            # Plot the trajectory
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot(path[0], path[1], path[2], label='Trajectory')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"{self.featureDetector} Trajectory")
            ax.legend()
            ax.text( path[0, 0],  path[1, 0],  path[2, 0], "Start")
            ax.text(path[0, -1], path[1, -1], path[2, -1],   "End")
            plt.show()

            # Save the Plot to Image File in Logs
            fig.savefig(plotFilename)
        
        logFile.close()
        resultsFile.close()

        return path

    def readTrajectory(self, runNumber: int):
        logFiles = os.listdir(self.resultsDir)
        filename = [os.path.join(self.resultsDir,x) for x in logFiles if f"_{runNumber}." in x][0]
        trajectory = np.array(pd.read_csv(filepath_or_buffer=filename, header=None, sep=","))
        return trajectory

