import cv2
import numpy as np

import matplotlib.pyplot as plt

class reprojectTo3D():
    def __init__(self, disparityMap = None):
        self.disparityMap = disparityMap
        self.stereo = None
        

    def reproject(self,img1, img2, img1Intrinsic:dict, img2Intrinsic:dict, extrinsic:dict, mtxKey:str="mtx", distKey:str="dist"):
        
        reImg1, reImg2, Q = self.makeStereoRectifyImg(img1=img1, img2=img2,img1Intrinsic=img1Intrinsic, img2Intrinsic=img2Intrinsic,
                                                    extrinsic=extrinsic, mtxKey=mtxKey, distKey=distKey)
        
        """
        #reprojectImageTo3Dで取得したZ軸情報を書き出し
        plt.imshow(reImg1)
        plt.colorbar()
        plt.plasma()
        #plt.savefig(data_dir+"/Depth_map/" + os.path.basename(file), format="png")
        plt.show()
        """
        
        self.makeDisparityMap(reImg1, reImg2)
        
        points_3d = cv2.reprojectImageTo3D(disparity=self.disparityMap, Q=Q)
        
        return points_3d


    #cv2.StereoSGBM(OpenCV2)->cv2.StereoSGBM_create(OpenCV3)
    def makeDisparityMap(self,  img1, img2):
        #入力は並行ステレオ化画像を想定
        if self.stereo == None:
            self.setStereoParameter()
        
        self.disparityMap = self.stereo.compute(img1,img2).astype(np.float32)/16.0
        cv2.imwrite('./reprojectTo3D/result/disparityMap.png', self.disparityMap)
        
    def setStereoParameter(self,minDisparity:int=0, numDisparities:int=96, blockSize:int=16, windowSize:int=3, p1:int=8, p2:int=32, 
                            disp12MaxDiff:int=1, uniquenessRatio:int=10, speckleWindowSize:int=100, speckleRange:int=32):    
        
        self.stereo = cv2.StereoSGBM_create(  minDisparity = minDisparity,
                                                    numDisparities = numDisparities,
                                                    blockSize = blockSize,
                                                    P1 = p1 * 3 * windowSize*windowSize, #The larger the values are, the smoother the disparity is. P1 is the penalty on the disparity change by plus or minus 1 between neighbor pixels. 
                                                    P2 = p2 * 3 * windowSize*windowSize, #P2 is the penalty on the disparity change by more than 1 between neighbor pixels. The algorithm requires P2 > P1 .
                                                    disp12MaxDiff=disp12MaxDiff,
                                                    uniquenessRatio = uniquenessRatio,
                                                    speckleWindowSize=speckleWindowSize,
                                                    speckleRange=speckleRange
                                                    )
        
    def makeStereoRectifyImg(self, img1, img2, img1Intrinsic:dict, img2Intrinsic:dict, extrinsic:dict, mtxKey:str="mtx", distKey:str="dist"):
        img_size = img1.T.shape
        
        mat1, dist1 = img1Intrinsic[mtxKey], img1Intrinsic[distKey]
        mat2, dist2 = img2Intrinsic[mtxKey], img2Intrinsic[distKey]
        R, T = extrinsic["R"], extrinsic["T"]
        
        R1, R2, P1, P2, Q, validPixROT1, validPixROI2 = cv2.stereoRectify(cameraMatrix1=mat1,distCoeffs1=dist1, cameraMatrix2=mat2,distCoeffs2=dist2, imageSize=img_size, R=R, T=T)
        
        #img1 rectify
        map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix=mat1,distCoeffs=dist1,
                                                    R=R1,newCameraMatrix=P1, size=img_size, m1type=cv2.CV_32FC1)
        reImg1 = cv2.remap(src=img1, map1=map1, map2=map2,interpolation=cv2.INTER_NEAREST)
        
        #img2 rectify
        map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix=mat2,distCoeffs=dist2,
                                                    R=R2,newCameraMatrix=P2, size=img_size, m1type=cv2.CV_32FC1)
        reImg2 = cv2.remap(src=img2, map1=map1, map2=map2,interpolation=cv2.INTER_NEAREST)

        return reImg1, reImg2, Q