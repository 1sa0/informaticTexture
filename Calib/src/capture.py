import cv2
import numpy as np
import os


def capture():
    capR = cv2.VideoCapture(3) #3
    capL = cv2.VideoCapture(0) #0
    
    while True:
        
        retR,frameR = capR.read()
        retL,frameL = capL.read()
        
        if not retR:
            print("Ignoring empty Right camera frame")
            continue
        
        if not retL:
            print("Ignoring empty Left camera frame")
            continue
        
        mergeImg = np.vstack((frameR,frameL))
        cv2.imshow("CamR/CamL", mergeImg)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
        elif cv2.waitKey(5) & 0xFF == 13:
            file_num = sum(os.path.isfile(os.path.join('./img/CalibImg', name)) for name in os.listdir('./img/CalibImg'))
            cv2.imwrite('img/CalibImg/calib_{}.png'.format(file_num + 1), mergeImg)
        
    capR.release()
    capL.release()
    
    
if __name__ == "__main__":
    capture()