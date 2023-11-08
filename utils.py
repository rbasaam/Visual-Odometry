import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

errorMsgs = {
    "err_DescriptorLength": "Descriptor length is less than 2, the size of the kd-tree cannot be less than 2.",
    "err_NoneDescriptor": "One of the descriptors is None, cannot compute matches."
}

class dataManager():
    def __init__(self, dataDir: str):
        self.rgbList = [os.path.join(os.path.join(dataDir, "pov"), rgbFile) for rgbFile in os.listdir(os.path.join(dataDir, "pov"))]
        self.depthList = [os.path.join(os.path.join(dataDir, "depth"), depthFile) for depthFile in os.listdir(os.path.join(dataDir, "depth"))]
        self.numImages = len(self.rgbList)

        self.grayImages = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in self.rgbList]
        self.depthImages = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in self.depthList]
    
    @staticmethod
    def _form_transf(R, t):

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T

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
    
    def getMatches(self, frameNum: int, featureDetector: str = "ORB"):

        
        if featureDetector == "ORB":
            # Initiate ORB detector
            detector = cv2.ORB_create(3000)
            # Match descriptors.
            flannIndex_LSH = 6
            index_params = dict(algorithm=flannIndex_LSH, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)

        elif featureDetector == "SIFT":
            # Initiate SIFT detector
            detector = cv2.SIFT_create(3000)
            # Match descriptors.
            index_params = dict(algorithm=0, trees=20)
            search_params = dict(checks=150)

        # Find the keypoints and descriptors with Detector
        kp1, des1 = detector.detectAndCompute(self.grayImages[frameNum], None)
        kp2, des2 = detector.detectAndCompute(self.grayImages[frameNum+1], None)
        # Create FLANN matcher
        flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

        if des1 is None or des2 is None:
            matches = 'err_NoneDescriptor'
        elif len(des1) < 2 or len(des2) < 2:
            matches = 'err_DescriptorLength'
        else:
            matches = flann.knnMatch(des1, des2, k=2)
    
        if matches in errorMsgs.keys():
            
            print("----------------------------Bad Boy----------------------------")
            print(f"{errorMsgs['err_DescriptorLength']} at Frame {frameNum}.")

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

            img3 = cv2.drawMatches(self.grayImages[frameNum+1], kp1, self.grayImages[frameNum],kp2, goodMatches ,None,**draw_params)
            cv2.imshow("image", img3)
            cv2.waitKey(200)

            # Get the image points form the good matches
            q1 = np.float32([kp1[m.queryIdx].pt for m in goodMatches])
            q2 = np.float32([kp2[m.trainIdx].pt for m in goodMatches])

        return q1, q2
    
    def getPose(self, q1, q2):
        # Get the essential matrix
        E, mask = cv2.findEssentialMat(q1, q2, method=cv2.RANSAC, prob=0.999, threshold=0.1)
        _, R, t, mask = cv2.recoverPose(E, q1, q2)
        T = self._form_transf(R, t)
        return T
        
    def plotTrajectory(self, featureDetector: str = "ORB"):
        trajectory = np.zeros((3, self.numImages))
        for i in range(1, self.numImages):
            q1, q2 = self.getMatches(frameNum=i, featureDetector=featureDetector)
            numMatches = q1.shape[0]
            if numMatches < 5:
                trajectory[:, i] = trajectory[:, i-1]
            else:
                transf = self.getPose(q1, q2)
                trajectory[:, i] = transf[:3, 3]

        # Plot the trajectory
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(trajectory[0], trajectory[1], trajectory[2], label='Trajectory')
        ax.legend()
        ax.text(trajectory[0, 0], trajectory[1, 0], trajectory[2, 0], "Start")
        ax.text(trajectory[0, -1], trajectory[1, -1], trajectory[2, -1], "End")
        plt.show()

        return trajectory