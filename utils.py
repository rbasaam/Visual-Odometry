import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging

# Define the Logging Configuration Messages
logging.basicConfig(
    filename="logs/runtime.log",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%b-%d %H:%M:%S",
)

# Define Custom Error Messages
errorMsgs = {
    "err_DescriptorLength": "Descriptor length is less than 2, the size of the kd-tree cannot be less than 2.\n",
    "err_NoneDescriptor": "One of the descriptors is None, cannot compute matches.\n",
    "err_InsufficientMatches": "Number of matches is less than 6, cannot compute Essential Matrix.\n"
}

# Define the DataManager Class
class dataManager():
    def __init__(self, rootDir: str):
        # Define Relevant Directories
        self.dataDir = os.path.join(rootDir, "data")
        self.logDir = os.path.join(rootDir, "logs")
        self.resultsDir = os.path.join(self.logDir, "results")
        self.plotDir = os.path.join(self.logDir, "plots")
        self.matchesDir = os.path.join(self.logDir, "matches")
        # Get the List of Image Filenames to Read
        self.rgbList = [os.path.join(os.path.join(self.dataDir, "pov"), rgbFile) for rgbFile in os.listdir(os.path.join(self.dataDir, "pov"))]
        self.depthList = [os.path.join(os.path.join(self.dataDir, "depth"), depthFile) for depthFile in os.listdir(os.path.join(self.dataDir, "depth"))]

        # Get the Number of Frames in the Dataset
        self.numImages = len(self.rgbList)

    def _loadImages(self):
        """
        Load the Images from the Dataset

        Returns:
            rgbImages: List of RGB Images
            grayImages: List of Grayscale Images
            depthImages: List of Depth Images
        """
        self.rgbImages = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in self.rgbList]
        self.grayImages = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in self.rgbList]
        self.depthImages = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in self.depthList]
    
    def _sampleFrame(self, frameNum: int):
        """
        Sample a Frame from the Dataset

        Args:
            frameNum (int): Frame Number to Sample
        """
        
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
    
    def _animateFrames(self, fps=20.0, outputFilename = None):
        """
        Animate the Frames from the Dataset

        Args:
            fps (float, optional): Frames per Second. Defaults to 20.0.
            outputFilename (str, optional): Output Filename. Defaults to "frameAnimation.mp4" if None.
        """
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
    
    def _createFeatureDetector(self, featureDetector="ORB"):
        """
        Create the Feature Detector

        Args:
            featureDetector (str, optional): Feature Detector to Use ("ORB"/"SIFT"). Defaults to "ORB".

        Returns:
            detector: Feature Detector Object
            index_params: Index Parameters for FLANN Matcher
            search_params: Search Parameters for FLANN Matcher
        """
        if featureDetector == "ORB":
            # Initiate ORB detector
            detector = cv2.ORB_create(3000)
            # Match descriptors.
            flannIndex_LSH = 6
            index_params = dict(
                algorithm=flannIndex_LSH, 
                table_number=6, 
                key_size=12, 
                multi_probe_level=1
                )
            search_params = dict(checks=50)

        elif featureDetector == "SIFT":
            # Initiate SIFT detector
            detector = cv2.SIFT_create(3000)
            # Match descriptors.
            index_params = dict(algorithm=0, trees=20)
            search_params = dict(checks=150)
        
        return detector, index_params, search_params

    def _readTrajectory(self, runNumber: int):
        """
        Read the Trajectory from the Results Folder

        Args:
            runNumber (int): Run Number of the Trajectory to Read

        Returns:
            trajectory: Trajectory Array
        """
        logFiles = os.listdir(self.resultsDir)
        filename = [os.path.join(self.resultsDir,x) for x in logFiles if f"_{runNumber}." in x][0]
        trajectory = np.array(pd.read_csv(filepath_or_buffer=filename, header=None, sep=","))
        return trajectory
       
    def _trackMotion(self, featureDetector="ORB", saveMatches=False):
        """
        Track the Motion of the Camera

        Args:
            featureDetector (str, optional): Feature Detector to Use ("ORB"/"SIFT"). Defaults to "ORB".
            saveMatches (bool, optional): Save the Matches to Image File. Defaults to False.

        Returns:
            path: Trajectory Array
        """
        # Load the Images
        logging.debug(f"Loading Images")
        self._loadImages()
        logging.info(f"Number of Images: {self.numImages}")
        # Determine and Log the Run Number
        runNumber = len(os.listdir(self.plotDir))+1
        logging.info(f"Run Number: {runNumber}")

        # Create and Log the Feature Detector
        detector, index_params, search_params = self._createFeatureDetector(featureDetector)
        logging.info(f"Feature Detector: {featureDetector}")
        
        # Initialize the Trajectory Array
        trajectory = np.zeros((3, self.numImages))
        logging.info(f"Trajectory Shape: {trajectory.shape}")

        # Detect and Match Features Between Frames
        for frameNum in range(1, self.numImages):
            logging.info(f"Detecting Matches Between Frame {frameNum} and Frame {frameNum+1}")
            # Find the keypoints and descriptors with Detector
            kp1, des1 = detector.detectAndCompute(self.grayImages[frameNum-1], None)
            kp2, des2 = detector.detectAndCompute(self.grayImages[frameNum], None)

            # Create FLANN matcher
            flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

            # Check if the descriptors are None or have length less than 2
            if des1 is None or des2 is None:
                matches = 'err_NoneDescriptor'
            elif len(des1) < 2 or len(des2) < 2:
                matches = 'err_DescriptorLength'
            else:
                matches = flann.knnMatch(des1, des2, k=2)
            # Error Handling and Flow Control
            if matches in errorMsgs.keys():
                logging.error(errorMsgs[matches])
                # Empty Mathces and Image Points Arrays                   
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
                # Draw Matches
                draw_params = dict(
                    matchColor = -1, # draw matches in green color
                    singlePointColor = None,
                    matchesMask = None, # draw only inliers
                    flags = 2
                )
                # Get the image points form the good matches
                q1 = np.float32([kp1[m.queryIdx].pt for m in goodMatches])
                q2 = np.float32([kp2[m.trainIdx].pt for m in goodMatches])

                # Draw the Matches
                img3 = cv2.drawMatches(self.grayImages[frameNum-1], kp1, self.grayImages[frameNum],kp2, goodMatches ,None,**draw_params)
                cv2.imshow("image", img3)
                if saveMatches:
                    # Save the Matches to Image File in Logs Folder
                    if not os.path.exists(os.path.join(self.matchesDir,f"{featureDetector}_{runNumber}")):
                        os.makedirs(os.path.join(self.matchesDir,f"{featureDetector}_{runNumber}"))
                    cv2.imwrite(os.path.join(os.path.join(self.matchesDir,f"{featureDetector}_{runNumber}"), f"matches_{frameNum}_{frameNum+1}.png"), img3)
                cv2.waitKey(200)

            # Estimate the Essential Matrix and Recover Pose
            numMatches = q1.shape[0]
            failedToMatch = 0
            if numMatches < 6:
                failedToMatch += 1
                trajectory[:, frameNum] = trajectory[:, frameNum-1]
                logging.debug(f"Number of Matches with Frame {frameNum+1}: {numMatches}\n")
                logging.warning(errorMsgs["err_InsufficientMatches"])
            else:
                E, mask = cv2.findEssentialMat(q1, q2, method=cv2.RANSAC, prob=0.999, threshold=0.1)
                _, R, t, mask = cv2.recoverPose(E, q1, q2)
                trajectory[:, frameNum] = t.flatten()
                logging.info("----------------------------------------------------------------------------------\n")
                logging.info(f"Number of Good Matches with Frame {frameNum+1}: {len(goodMatches)}\n")
                logging.info(f"Rotation Matrix Shape {R.shape}\n")
                logging.info(f"Rotation Matrix \n {R} \n")
                logging.info(f"Translation Matrix Shape {t.shape}\n")
                logging.info(f"Translation Matrix \n {t}\n")
                logging.debug(f"Number of Inliers with Frame {frameNum+1}: {np.sum(mask)}\n")
                logging.debug(f"Essential Matrix Shape {E.shape}\n")
                logging.debug(f"Essential Matrix \n {E} \n")
                logging.info("----------------------------------------------------------------------------------\n")
        logging.debug(f"Number of Frames with Insufficient Matches: {failedToMatch}\n")
        
        # Compute the Path by Integrating Trajectories
        path = np.cumsum(trajectory, axis=1)
                
        # Save the Trajectory to Text File in Logs Folder
        resultsFilename = os.path.join(self.resultsDir, f"{featureDetector}_path_{runNumber}.txt")
        with open(resultsFilename, "a") as resultsFile:
            np.savetxt(resultsFile, path.transpose(), delimiter=",", newline="\n", fmt="%.4f")
        
        # Plot and Save the Trajectory to Image File in Logs Folder
        plotFilename = os.path.join(self.plotDir, f"{featureDetector}_trajectory_{runNumber}.png")
        # Plot the trajectory
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(path[0], path[1], path[2], label='Trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{featureDetector} Trajectory")
        ax.legend()
        ax.text( path[0, 0],  path[1, 0],  path[2, 0], "Start")
        ax.text(path[0, -1], path[1, -1], path[2, -1],   "End")
        plt.show()

        # Save the Plot to Image File in Logs
        fig.savefig(plotFilename)
        
        return path
