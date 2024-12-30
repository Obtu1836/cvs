import cv2 
import numpy as np 

''''
左键画线段 右键画圆

设定self.imk和 self.img.copy()是为了动态显示 只保留当前的操作结果 不进行累积画图

'''

class Paint:
    def __init__(self,path):

        self.img=cv2.imread(path)
        self.imk=self.img.copy()

        self.n=0
    
    def fun(self,event,x,y,flag,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            self.points.append((x,y))
            if len(self.points)>=1:
                self.imk=cv2.line(self.img.copy(),self.points[self.n-1],self.points[self.n],(0,0,255),2)
            self.n+=1

        elif event==cv2.EVENT_RBUTTONDOWN:
            self.imk=cv2.circle(self.img.copy(),(x,y),5,(0,255,0),2)

    def run(self):
       self.points=[]  # 主运行函数  1设定窗口 2设定鼠标操作 3动态显示图片 
       cv2.namedWindow('img')
       cv2.setMouseCallback('img',self.fun)
       self.show('img')

    def show(self,name):

        while True:
            cv2.imshow(name,self.imk)
            key=cv2.waitKey(10)
            if key==ord('x'):
                break
        cv2.destroyWindow(name)

if __name__ == '__main__':
    path=r'./imgs/sea.jpg'
    p=Paint(path)
    p.run()
