import cv2
import numpy as np

def makeDisparityMap(imgR, imgL):
    window_size = 3
    min_disparity = 0
    num_disparity = 96
    block_size = 16
    
    stereo = cv2.StereoSGBM_create(minDisparity = min_disparity,
                                    numDisparities = num_disparity,
                                    blockSize = block_size,
                                    P1 = 8 * 3 * window_size * window_size, #視差の滑らかさを制御するパラメタ
                                    P2 = 32* 3 * window_size * window_size, #同上
                                    disp12MaxDiff = 1, 
                                    uniquenessRatio = 10,
                                    speckleWindowSize = 100,
                                    speckleRange = 32
                                )
    
    disparity_map = stereo.compute(imgR,imgL).astype(np.float32)/16.0
    
    return disparity_map
    
if __name__ == "__main__":
    imgR = cv2.imread('./3dimReprojection/result/rectifiedImgR.png')
    imgL = cv2.imread('./3dimReprojection/result/rectifiedImgL.png')
    disparity_map = makeDisparityMap(imgR=imgR, imgL=imgL)
    cv2.imwrite('./3dimReprojection/result/disparityMap.png',disparity_map)