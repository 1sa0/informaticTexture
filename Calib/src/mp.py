import cv2
import yaml


def stereoRectify(camR_path:str = './Calib/result/intrinsic_R.yml',camL_path:str = './Calib/result/intrinsic_L.yml'):
    matR, distR = readYml(camR_path)
    matL, distL = readYml(camL_path)
    
    


def readYml(path):
    with open(path, 'r') as f:
        parameter = yaml.safe_load(f)
        mat = parameter["intrinsic"]
        dist = parameter["distortion"]
    
    return mat, dist

if __name__== "__main__":
    stereoRectify()
