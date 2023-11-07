import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class dataManager():
    def __init__(self, dataDir: str):
        self.rgbList = [os.path.join(os.path.join(dataDir, "rgb"), rgbFile) for rgbFile in os.listdir(os.path.join(dataDir, "rgb"))]
        self.depthList = [os.path.join(os.path.join(dataDir, "depth"), depthFile) for depthFile in os.listdir(os.path.join(dataDir, "depth"))]
        self.numImages = len(self.rgbList)

        self.grayImages = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in self.rgbList]
        self.depthImages = [np.loadtxt(img, dtype=np.float32, delimiter=',') for img in self.depthList]
    
    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T

    def sampleFrame(self, frameNum: int):
        
        rgbImage = cv2.cvtColor(cv2.imread(self.rgbList[frameNum-1]), cv2.COLOR_BGR2RGB)
        grayImage = cv2.imread(self.rgbList[frameNum-1], cv2.IMREAD_GRAYSCALE)
        depthImage = np.loadtxt(self.depthList[frameNum-1], dtype=np.float32, delimiter=',')

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(rgbImage)
        ax[0].set_title(f"Frame {frameNum} RGB") 
        ax[1].imshow(grayImage, cmap='gray')
        ax[1].set_title(f"Frame {frameNum} Grayscale")
        ax[2].imshow(depthImage, cmap='gray')
        ax[2].set_title(f"Frame {frameNum} Depth")
        plt.show()
        return
    
    def getMatches(self, frameNum: int):

        # Initiate ORB detector
        orb = cv2.ORB_create(3000)

        # Find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(self.grayImages[frameNum], None)
        kp2, des2 = orb.detectAndCompute(self.grayImages[frameNum+1], None)

        # Match descriptors.
        flannIndex_LSH = 6
        index_params = dict(algorithm=flannIndex_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        matches = flann.knnMatch(des1, des2, k=2)

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
        
