import cv2 
import numpy as np 
from numpy.linalg import norm

class Sony:
    def __init__(self,path1,path2):

        self.fg=cv2.imread(path1)
        self.bg=cv2.imread(path2)

        self.fg_point=[]
        self.bg_point=[]

    def fun(self,event,x,y,flag,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            if param==0:
                cv2.circle(self.fg,(x,y),2,(0,255,0),2)
                self.fg_point.append((x,y))
            else:
                cv2.circle(self.bg,(x,y),2,(0,0,255),2)
                self.bg_point.append((x,y))

    def make_window(self,name,img,param):
        cv2.namedWindow(name,cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(name,self.fun,param)
        while True:
            cv2.imshow(name,img)
            key=cv2.waitKey(10)
            if key==ord('x'):
                break
        cv2.destroyWindow(name)

    
    def run(self):
        ims=self.fg
        self.make_window('fg',ims,0)
        
        self.fg_point=np.float32(self.fg_point)
        h,w=self.cal_wh(self.fg_point)
        corner=np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
        m1=cv2.getPerspectiveTransform(self.fg_point,corner)
        mid_img=cv2.warpPerspective(self.fg,m1,(w,h))

        ims=self.bg
        self.make_window('bg',ims,1)
        self.bg_point=np.float32(self.bg_point).reshape(-1,1,2)
        m2=cv2.getPerspectiveTransform(corner,self.bg_point)
        h2,w2=self.bg.shape[:2]
        sf=cv2.warpPerspective(mid_img,m2,(w2,h2))

        px=cv2.fillPoly(self.bg,[self.bg_point.astype(np.int32)],(0,0,0))
        res=cv2.bitwise_or(px,sf)
        cv2.imshow('res',res)
        cv2.waitKey(0)


    def cal_wh(self,rect):
        p1,p2,p3,p4=rect
        h=int(max(norm(p2-p1),norm(p4-p3)))
        w=int(max(norm(p3-p2),norm(p4-p1)))
        return h,w

if __name__ == '__main__':
    path1=r'imgs/qizi.jpeg'
    path2=r'imgs/demo2.jpg'

    s=Sony(path1,path2)
    s.run()
