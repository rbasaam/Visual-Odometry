import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from icecream import ic

errorFlags = [
    'err_noImage', 
    'err_noDescriptor', 
    'err_descriptorMismatch',
    'No Match'
]

class datasetHandler:
    def __init__(self, rootDir: str):
        self.rootDir = rootDir
        self.povDir = os.path.join(rootDir, "pov")
        self.depthDir = os.path.join(rootDir, "depth")

        self.imgList = sorted(os.listdir(self.povDir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.depthList = sorted(os.listdir(self.depthDir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.numImages = len(self.imgList)   

        self.rgbImages = []
        self.grayImages = []
        self.depthImages = []

        for img in self.imgList:
            # Read Images and Ensure they are in RGB and Grayscale
            self.rgbImages.append(cv2.cvtColor(cv2.imread(os.path.join(self.povDir, img)), cv2.COLOR_BGR2RGB))
            self.grayImages.append(cv2.cvtColor(cv2.imread(os.path.join(self.povDir, img)), cv2.COLOR_BGR2GRAY))
        for img in self.depthList:
            self.depthImages.append(cv2.cvtColor(cv2.imread(os.path.join(self.depthDir, img)), cv2.COLOR_BGR2GRAY))
        return
    
    def showImage(self, frame: int):
        """
        Plot a Subplot of the Image and its Grayscale Equivalent for a given index

        Arguments:
        frame -- frame number of the image to be plotted

        Returns:
        None
        """
        index = frame - 1
        fig, ax = plt.subplots(1, 3, figsize=(20, 10))
        ax[0].imshow(self.rgbImages[index])
        ax[0].set_title(f"RGB Image at Frame {frame}")
        ax[1].imshow(self.grayImages[index], cmap='gray')
        ax[1].set_title(f"Grayscale Image at Frame {frame}")
        ax[2].imshow(self.depthImages[index], cmap='gray')
        ax[2].set_title(f"Depth Image at Frame {frame}")
        plt.show()

        return
    
def extractFeatures(grayImage):
    """
    Extract features from the image

    Arguments:
    datasetHandler -- datasetHandler object
    index -- index of the image to be plotted

    Returns:
    kp -- keypoints
    des -- descriptors
    
    """
    # Create feature detector and descriptor extractor
    feature_detector = cv2.ORB_create()
    descriptor_extractor = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    kp, des = feature_detector.detectAndCompute(grayImage, None)
    print(f"Number of features detected in frame: {len(kp)}\n")
    
    return kp, des

def drawKeypoints(grayImage): 
    """
    Plot the keypoints on the image

    Arguments:
    datasetHandler -- datasetHandler object
    index -- index of the image to be plotted

    Returns:
    None

    """
    # Create feature detector and descriptor extractor
    feature_detector = cv2.ORB_create()
    descriptor_extractor = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    kp, des = feature_detector.detectAndCompute(grayImage, None)

    # Plot the keypoints on the image
    img = cv2.drawKeypoints(grayImage, kp, None, color=(0, 255, 0), flags=0)
    plt.figure(figsize=(20, 10))
    plt.imshow(img)
    plt.show()

    return
    
def matchFeatures(datasetHandler: datasetHandler, frameNumber: int, filter=None, summarize=False):
    """
    Match features between two images

    Arguments:
    image1 -- first image
    image2 -- second image

    Returns:
    matches -- list of matches between the two images

    """
    # Create feature detector and descriptor extractor
    feature_detector = cv2.ORB_create()
    descriptor_extractor = cv2.ORB_create()

    image1 = datasetHandler.grayImages[frameNumber]
    image2 = datasetHandler.grayImages[frameNumber+1]

    # Detect keypoints and compute descriptors
    kp1, des1 = feature_detector.detectAndCompute(image1, None)
    kp2, des2 = feature_detector.detectAndCompute(image2, None)

    # Check if images are None or descriptors have different types or shapes
    if image1 is None or image2 is None:
        print(f"Image at frame {frameNumber} or {frameNumber+1} is None")
        matches = 'err_noImage'
    elif des1 is None or des2 is None:
        print(f"Descriptor at frame {frameNumber} or {frameNumber+1} is None")
        matches = 'err_noDescriptor'
    elif des1.dtype != des2.dtype or des1.shape[1] != des2.shape[1]:
        print(f"Descriptors at frame {frameNumber} and {frameNumber+1} have different types or shapes")
        matches = 'err_descriptorMismatch'
    else:
        # Create a Brute Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance
        matches = sorted(matches, key=lambda x: x.distance)
        numMatches = len(matches)

        if summarize:
            # Draw first 10 matches
            img = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], None, flags=2)
            plt.figure(figsize=(20, 10))
            plt.imshow(img)
            plt.show()

            for i, match in enumerate(matches):
                print(f"Match {i}:")
                print(f"Image 1 descriptor index: {match.queryIdx}")
                print(f"Image 2 descriptor index: {match.trainIdx}")
                print(f"Distance: {match.distance}")
                print()

        if filter is None:
            print(f"Matches Found: {numMatches}")
        elif filter is not None:
            maxDistance = max([match.distance for match in matches])
            matches = [match for match in matches if match.distance <= filter * maxDistance]
            print(f"Matches Found After Filtering: {len(matches)}/{numMatches}")
        
    return matches


def estimateMotion(matches, keypoints, descriptors, frameNumber: int):
    """
    Estimate the Motion Between Stream of Images

    Arguments:
    matches -- list of n-1 matches between n images
    keypoints -- list of keypoints for n images
    descriptors -- list of descriptors for n images
    frameNumber -- frame number of the image to be analyzed

    Returns:

    R -- rotation matrix
    t -- translation vector

    """

    # Get the matches for the current frame
    matches = matches[frameNumber]

    # Get the keypoints for the current frame
    kp1 = keypoints[frameNumber]
    kp2 = keypoints[frameNumber+1]

    # Get the descriptors for the current frame
    des1 = descriptors[frameNumber]
    des2 = descriptors[frameNumber+1]

    # Get the coordinates of the matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate the essential matrix
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Recover the pose from the essential matrix
    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts)

    return R, t

def animateFrames(fileList, outputFilename, fps=20):
    """
    Animate the frames

    Arguments:
    fileList -- list of filenames of the frames
    outputFilename -- filename of the output video
    fps -- frames per second

    Returns:
    None

    """
    # Get frame size
    img0 = cv2.imread(fileList[0])
    height, width, _ = img0.shape
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(outputFilename, fourcc, fps, (width, height))

    # Iterate through the frames
    for filename in fileList:
        # Read the frame
        img = cv2.imread(filename)

        # Write the frame to the video
        video.write(img)

    # Release the VideoWriter object
    video.release()

    return