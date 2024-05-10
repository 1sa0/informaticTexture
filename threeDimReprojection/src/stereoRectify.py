import cv2
import numpy as np
import yaml
import os

def makeStereoRectifyImg(camR_filename:str = 'intrinsic_R.yml', camL_filename:str = 'intrinsic_L.yml', ex_filename:str = 'extrinsic.yml'):
    matR, distR = readIntrinsicYml(camR_filename)
    matL, distL = readIntrinsicYml(camL_filename)
    E, F, R, T = readExtrinsicYml(ex_filename)

    img = cv2.imread('./3dimReprojection/img/calib_1.png', cv2.IMREAD_GRAYSCALE)
    imgR = img[:img.shape[0]//2,:]
    imgL = img[img.shape[0]//2:,:]
    img_size = (imgL.shape[1],imgL.shape[0])

    R1,R2,P1,P2,Q,validPixROI1,validPixROI2 = cv2.stereoRectify(matR,distR,matL,distL,img_size,R,T)

    map1_R, map2_R = cv2.initUndistortRectifyMap(matR,distR,R1,P1, img_size,cv2.CV_32FC1)
    map1_L, map2_L = cv2.initUndistortRectifyMap(matL,distL,R2,P2, img_size,cv2.CV_32FC1)
    
    ReImgR = cv2.remap(imgR, map1_R, map2_R, cv2.INTER_NEAREST)
    ReImgL = cv2.remap(imgL, map1_L, map2_L, cv2.INTER_NEAREST)

    return ReImgR, ReImgL

def readIntrinsicYml(filename:str='intrinsic.yml', path:str='./Calib/result'):
    fs = cv2.FileStorage(os.path.join(path, filename), cv2.FILE_STORAGE_READ)    
    mtx = fs.getNode("intrinsic").mat()
    dist = fs.getNode("distortion").mat()
    fs.release()
    return mtx, dist

def readExtrinsicYml(filename:str='extrinsic.yml', path:str='./Calib/result'):
    fs = cv2.FileStorage(os.path.join(path, filename), cv2.FILE_STORAGE_READ)
    E = fs.getNode("E").mat()
    F = fs.getNode("F").mat()
    R = fs.getNode("R").mat()
    T = fs.getNode("T").mat()
    fs.release()
    return E, F, R, T

if __name__== "__main__":
    imgR, imgL = makeStereoRectifyImg()
    
    cv2.imwrite('./StereoRectify/result/rectifiedImgR.png',imgR)
    cv2.imwrite('./StereoRectify/result/rectifiedImgL.png',imgL)