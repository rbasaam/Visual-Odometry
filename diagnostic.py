from utils import *
from icecream import ic


matchesFolder = "S:\\GitHub\\Visual-Odometry\\logs\\matches"
siftFolder = os.path.join(matchesFolder, "SIFT_1")
orbFolder = os.path.join(matchesFolder, "ORB_2")

def animateFrames(folder, fps=20.0):
    
    outputFilename = f"{folder}.mp4"
    imgList = [os.path.join(folder, x) for x in os.listdir(folder)]
    imgList.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    ic(imgList)
    numImages = len(imgList)

    height, width = cv2.imread(imgList[0], cv2.IMREAD_GRAYSCALE).shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(outputFilename, fourcc, fps, (width, height))
    for f in range(numImages):
        frame =cv2.imread(imgList[f], cv2.IMREAD_COLOR)
        video.write(frame)
    video.release()
    return

animateFrames(siftFolder)
animateFrames(orbFolder)