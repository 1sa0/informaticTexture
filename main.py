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
    #plt.savefig(data_dir+"/Depth_map/" + os.path.basename(file), format="png")
    plt.show()
    
    mask = disparity_map > disparity_map.min()

    output_points = np.zeros_like(disparity_map)
    #視差が取れない外れ値を0で置換
    output_points = points_3d * np.stack([mask]*3, axis=2)
    output_points[output_points == np.inf] = np.nan
    output_points[output_points == -np.inf] = np.nan

    """
    #reprojectImageTo3Dで取得したZ軸情報を書き出し
    plt.imshow(output_points[:,:,2],vmin = 0,vmax=800)
    plt.colorbar()
    plt.plasma()
    #plt.savefig(data_dir+"/Depth_map/" + os.path.basename(file), format="png")
    plt.show()
    """

    

if __name__ == "__main__":
    camLeftID = 2
    camRightID = 3 
    main(camRightID, camLeftID)