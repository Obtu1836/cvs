import cv2 
import numpy as np 

'''
简化版本
'''

class Grabcut:
    def __init__(self,path):

        self.img=cv2.imread(path)
        self.ims=self.img.copy()

        self.mask=np.zeros(self.img.shape[:2],dtype=np.uint8)
        self.bg=np.zeros((1,65),dtype=np.float64)
        self.fg=np.zeros((1,65),dtype=np.float64)

    def rect(self):

        cv2.namedWindow('rect',cv2.WINDOW_AUTOSIZE)
        x,y,w,h=cv2.selectROI('rect',self.ims)
        rect=(x,y,x+w,y+h)
        cv2.grabCut(self.ims,self.mask,rect,self.bg,self.fg,iterCount=1,
                    mode=cv2.GC_INIT_WITH_RECT)
        
    def make_mask(self,name):

        cv2.namedWindow(name,cv2.WINDOW_AUTOSIZE)
        x,y,w,h=cv2.selectROI(name,self.ims)
        if name=='bg':
            self.mask[y:y+h,x:x+w]=cv2.GC_PR_BGD
        else:
            self.mask[y:y+h,x:x+w]=cv2.GC_PR_FGD

        cv2.destroyWindow(name)
        cv2.grabCut(self.ims,self.mask,None,self.bg,self.fg,iterCount=1,
                    mode=cv2.GC_INIT_WITH_MASK)
        
    def run(self):

        self.rect()

        while True:
            mask1=np.where((self.mask==0)|(self.mask==2),0,1)
            mask1=mask1.astype(np.uint8)
            self.img=self.ims*mask1[:,:,None]
            cv2.imshow('ims',self.ims)
            cv2.imshow('img',self.img)

            key=cv2.waitKey(1)
            if key==ord('x'):
                break
            elif key==ord('b'):
                self.make_mask('bg')
            elif key==ord('f'):
                self.make_mask('fg')
        cv2.destroyAllWindows()

if __name__ == '__main__':
    path=r'/Users/yan/Books/OpenCV 4.5/ch13(ok)/13.3/kt.jpg'
    g=Grabcut(path)
    g.run()
