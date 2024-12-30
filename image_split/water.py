import cv2 
import numpy as np 

class Water:
    def __init__(self,path):

        self.img=cv2.imread(path)

    def run(self,):
        gray=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        _,thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        open=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
        sure_bg=cv2.dilate(open,kernel,iterations=2)
        dist=cv2.distanceTransform(open,cv2.DIST_L2,5)
        _,sure_fg=cv2.threshold(dist,dist.max()*0.7,255,cv2.THRESH_BINARY)
        sure_fg=sure_fg.astype(np.uint8)
        unknown=cv2.subtract(sure_bg,sure_fg)

        _,marker=cv2.connectedComponents(sure_fg)

        marker+=1
        marker[unknown==255]=0

        cv2.watershed(self.img,marker)
        marker[[0,-1],:]=0
        marker[:,[0,-1]]=0
        
        self.img[marker==-1]=(0,0,255)

        self.show(self.img)
        
    def show(self,img):
        cv2.imshow('img',img)
        cv2.waitKey(0)

if __name__ == '__main__':
    path=r'./imgs/coin.png'
    w=Water(path)
    w.run()

        
