from Calib.calibration import Calib
#from threeDimReprojection.src import
#from handPoseEstimation.src import 

def main(camLeftID:int = 0 , camRightID:int = 0, ymlDir:str="./Calib/resultsYml"):
    calibration = Calib(ymlDir=ymlDir)
    cam1Intrinsic, cam2Intrinsic, extrinsic = calibration.getParameter()


if __name__ == "__main__":
    camLeftID = 2
    camRightID = 3 
    main(camRightID, camLeftID)