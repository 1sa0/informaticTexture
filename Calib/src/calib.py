import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import yaml
import os

#Defining the dim of checkerboard
CHECKER_BOARD = (7,10)


def CalibrateCamera(img_path):
    
    #cv::TermCriteria::TermCriteria	(int type, int maxCount, double epsilon)	
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #objp:チェスボード内コーナーの3D座標リスト用のidx，チェスボードは平面を仮定しz方向は0のみとする, [(0,0,0),(1,0,0)...(9,12,zn)]
    objp = np.zeros((CHECKER_BOARD[0]*CHECKER_BOARD[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKER_BOARD[0], 0:CHECKER_BOARD[1]].T.reshape(-1,2)
    
    #キャリブ画像すべてのコーナ座標を保存
    objL_points = [] #チェスボード座標系3次元点
    objR_points = []
    
    imgL_points = [] #キャリブ画像座標系2次元点
    imgR_points = []
    
    files = glob.glob(img_path)
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'Image loading failure:{file}')
            continue
        
        img_R,img_L = splitGrayImg(img)

        retR, cornersR = cv2.findChessboardCorners(img_R, CHECKER_BOARD, None)
        retL, cornersL = cv2.findChessboardCorners(img_L, CHECKER_BOARD, None)
        
        if retR == True & retL:
            objR_points.append(objp)
            objL_points.append(objp)
            
            cornersR2 = cv2.cornerSubPix(img_R, cornersR, (11,11),(-1,-1), criteria)
            cornersL2 = cv2.cornerSubPix(img_L, cornersL, (11,11),(-1,-1), criteria)

            imgR_points.append(cornersR2)
            imgL_points.append(cornersL2)
            
    if (len(objL_points)>0 and len(imgL_points)>0) and (len(objR_points)>0 and len(imgR_points)>0): 
        retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objR_points, imgR_points, img_R.shape[::-1], None, None)
        retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objL_points, imgL_points, img_L.shape[::-1], None, None)

    else:
        print("Calibration failure")


    retval, mtxR, distR, mtxL, distL, R, T, E, F = cv2.stereoCalibrate(objectPoints=objR_points, imagePoints1=imgR_points,imagePoints2=imgL_points,
                                    cameraMatrix1=mtxR, distCoeffs1=distR, cameraMatrix2=mtxL,distCoeffs2=distL,
                                    imageSize=img_R.shape,criteria=criteria,flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    
    writeIntrinsicYml(mtxR,distR, 'intrinsic_R.yml')
    writeIntrinsicYml(mtxL,distL, 'intrinsic_L.yml')
    writeExtrinsicYml(retval,R,T,E,F)
    
def splitGrayImg(img):
    img_R = img[:img.shape[0]//2,:]
    img_L = img[img.shape[0]//2:,:]
    return img_R, img_L


def writeIntrinsicYml(mtx, dist, filename:str='intrinsic.yml',path:str='./Calib/result'):
    fs = cv2.FileStorage(os.path.join(path,filename), flags=cv2.FILE_STORAGE_WRITE)
    fs.write('intrinsic', mtx)
    fs.write('distortion',dist)
    fs.release()

def writeExtrinsicYml(retval,R,T,E,F, filename:str='extrinsic.yml', path:str='./Calib/result'):
    fs = cv2.FileStorage(os.path.join(path, filename), flags=cv2.FILE_STORAGE_WRITE)
    fs.write('retval',retval)
    fs.write('R', R)
    fs.write('T', T)
    fs.write('E', E)
    fs.write('F', F)
    fs.release()

if __name__ == "__main__":
    img_path = "./Calib/img/calib_*.png"
    CalibrateCamera(img_path)