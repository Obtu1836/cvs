import cv2 
import numpy as np 

class Brisk:
    def __init__(self,path):

        self.img=cv2.imread(path)
        self.brisk=cv2.AKAZE.create()
        self.kp1,self.desc1=self.brisk.detectAndCompute(self.img,None)

    def draw(self):
        
        new=cv2.drawKeypoints(self.img,self.kp1,None,(0,0,255),
                          cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        cv2.imshow('img',new)
        cv2.waitKey(0)

if __name__ == '__main__':
    path=r'imgs/B11643_6_15.png'
    take=Brisk(path)
    take.draw()
