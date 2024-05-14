from Calib.calibration import Calib
from reprojectTo3D.reproject import reprojectTo3D
#from handPoseEstimation.src import 

import cv2
import matplotlib.pyplot as plt
import numpy as np

def main(camLeftID:int = 0 , camRightID:int = 0, ymlDir:str="./Calib/resultsYml"):
    calibration = Calib(ymlDir=ymlDir)
    cam1Intrinsic, cam2Intrinsic, extrinsic = calibration.getParameter()

    reproject = reprojectTo3D()
    
    img = cv2.imread('./Calib/img/calib_10.png', 0)
    img1 = img[:img.shape[0]//2,:]
    img2 = img[img.shape[0]//2:,:]
    
    points_3d = reproject.reproject(img1,img2, cam1Intrinsic, cam2Intrinsic, extrinsic)
    disparity_map = reproject.disparityMap
    
    plt.imshow(disparity_map)
    plt.colorbar()
    plt.plasma()
    plt.show()
    

if __name__ == "__main__":
    camLeftID = 2
    camRightID = 3 
    main(camRightID, camLeftID)