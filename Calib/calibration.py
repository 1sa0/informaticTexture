import yaml
import glob
import cv2
import numpy as np
import os


class Calib():
    def __init__(self, ymlDir:str, checker_board:tuple=(7,10)):
        self.ymlDir = ymlDir
        self._CHECKER_BOARD = checker_board
        
        #cv::TermCriteria::TermCriteria	(int type, int maxCount, double epsilon)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def getParameter(self, cam1YmlFilename:str='intrinsic1.yml', cam2YmlFilename:str='intrinsic2.yml', exYmlFilename:str='extrinsic.yml'):
        """
        Loads the internal and external parameters of two cameras from a YML file.
        
        Parameters:
        cam1YmlFilename(str), cam2YmlFilename(str), exYmlFilename (str): The YML filename containing the parameters.
            Default filenames are intrinsic1.yml, intrinsic2.yml and extrinsic.yml.
        
        Returns:
            cam1Intrinsic(dict), cam2Intrinsic(dict), extrinsic(dict): The camera parameters.
        """
        
        
        cam1Intrinsic = self._readIntrinsicYml(self.ymlDir + '/' + cam1YmlFilename)
        cam2Intrinsic = self._readIntrinsicYml(self.ymlDir + '/' + cam2YmlFilename)
        extrinsic = self._readExtrinsicYml(self.ymlDir + '/' + exYmlFilename)

        return cam1Intrinsic, cam2Intrinsic, extrinsic
    
    def writeYml(self, calibImgDir:str, outPutDir:str=None, mode:int=0, cam1YmlFilename:str='intrinsic1.yml', cam2YmlFilename:str='intrinsic2.yml', exYmlFilename:str='extrinsic.yml'):
        """ 
        Loads a calibration images, perform the calibration, and write the results to a YML file.
        
        Parameters:
        calib_img_dir (str): The path to the directory containing calibration images.
        output_path (str optional): The file path where the YML file containing the calibration results will be saved.  Default is Calib.ymlDir
        mode (int, optional): The type op concatenated calibration images.
                                0 indicates images concatenated vertically,
                                1 indicates images concatenated horizontally.
                                Default is 0.
        """
        if outPutDir == None:
            outPutDir = self.ymlDir
            
        self._setSplitMode(mode)
        
        # return val: cv2.stereoCalibrate()
        retval, mtx1, dist1, mtx2, dist2, R, T, E, F = self._calibrationProcess(calibImgDir)
        
        
        print(outPutDir)
        self._writeIntrinsicYml(mtx1,dist1, filename = cam1YmlFilename, path = outPutDir)
        self._writeIntrinsicYml(mtx2,dist2, filename = cam2YmlFilename, path = outPutDir)
        self._writeExtrinsicYml(retval,R,T,E,F, filename = exYmlFilename, path = outPutDir)

    ###
    ###ここからプライベートメソッド
    ###
    
    def _calibrationProcess(self, calibImgDir):
        
        #objp:チェスボード内コーナーの3D座標リスト用のidx，チェスボードは平面を仮定しz方向は0のみとする, [(0,0,0),(1,0,0)...(9,12,zn)]
        objp = np.zeros((self._CHECKER_BOARD[0]*self._CHECKER_BOARD[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:self._CHECKER_BOARD[0], 0:self._CHECKER_BOARD[1]].T.reshape(-1,2)
        
        #キャリブ画像すべてのコーナ座標を保存
        obj1_points = [] #チェスボード座標系3次元点
        obj2_points = []
        
        img1_points = [] #キャリブ画像座標系2次元点
        img2_points = []
        
        
        
        files = glob.glob(calibImgDir + "/*.png")
        for file in files:
            #ここから関数分けたい どうしよう
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f'Image loading failure:{file}')
                continue
            img1, img2 = self._splitImg(img)
            ret1, corners1 = cv2.findChessboardCorners(img1, self._CHECKER_BOARD, None)
            ret2, corners2 = cv2.findChessboardCorners(img2, self._CHECKER_BOARD, None)
            
            if ret1 == True & ret2:
                obj1_points.append(objp)
                obj2_points.append(objp)
                
                corners1 = cv2.cornerSubPix(img1, corners1, (11,11),(-1,-1), self.criteria)
                corners2 = cv2.cornerSubPix(img2, corners2, (11,11),(-1,-1), self.criteria)

                img1_points.append(corners1)
                img2_points.append(corners2)
            #ここまで
        
        if (len(obj1_points)>0 and len(img1_points)>0) and (len(obj2_points)>0 and len(img2_points)>0): 
            ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(obj1_points, img1_points, img1.shape[::-1], None, None)
            ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(obj2_points, img2_points, img2.shape[::-1], None, None)

        else:
            print("Calibration failure")
            return False


        retval, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(objectPoints=obj1_points, imagePoints1=img1_points, imagePoints2=img2_points,
                                        cameraMatrix1=mtx1, distCoeffs1=dist1, cameraMatrix2=mtx2, distCoeffs2=dist2,
                                        imageSize=img2.shape, criteria=self.criteria, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        
        return retval, mtx1, dist1, mtx2, dist2, R, T, E, F
        
    def _readIntrinsicYml(self, ymlPath:str):
        fs = cv2.FileStorage(ymlPath, cv2.FILE_STORAGE_READ)    
        mtx = fs.getNode("intrinsic").mat()
        dist = fs.getNode("distortion").mat()
        fs.release()
    
        return {"mtx":mtx, "dist":dist}
    
    
    def _readExtrinsicYml(self, ymlPath:str):
        fs = cv2.FileStorage(ymlPath, cv2.FILE_STORAGE_READ)
        E = fs.getNode("E").mat()
        F = fs.getNode("F").mat()
        R = fs.getNode("R").mat()
        T = fs.getNode("T").mat()
        fs.release()
        return {"E":E, "F":F, "R":R, "T":T}
    
    
    def _setSplitMode(self, mode:int=0):
        
        if mode == 0:
            self._splitImg = self._splitVertical
        else:
            self._splitImg = self._splitHorizontal
        
    def _splitVertical(self, img):
        img1 = img[:img.shape[0]//2,:]
        img2 = img[img.shape[0]//2:, :]
        return img1, img2
    
    def _splitHorizontal(self, img):
        img1 = img[:, :img.shape[1]//2]
        img2 = img[:, img.shape[1]//2:]
        return img1, img2
    
    def _writeIntrinsicYml(self, mtx, dist, path:str, filename:str="intrinsic.yml"):
        fs = cv2.FileStorage(os.path.join(path,filename), flags=cv2.FILE_STORAGE_WRITE)
        fs.write('intrinsic', mtx)
        fs.write('distortion',dist)
        fs.release()

    def _writeExtrinsicYml(self, retval,R,T,E,F, path:str, filename:str='extrinsic.yml'):
        fs = cv2.FileStorage(os.path.join(path, filename), flags=cv2.FILE_STORAGE_WRITE)
        fs.write('retval',retval)
        fs.write('R', R)
        fs.write('T', T)
        fs.write('E', E)
        fs.write('F', F)
        fs.release()
        
        
if __name__ == "__main__":
    calib = Calib(ymlDir='./Calib/resultsYml')
    calib.writeYml(calibImgDir="./Calib/img/")