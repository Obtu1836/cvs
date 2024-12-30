import cv2
import numpy as np 
'''
通过滑动条或者鼠标事件 show函数参数 不能传入对应的图像进行显示 

例如 不能进行 self.show(self,img)

同时还需要注意 如果cv2.imshow(*,img) 其中img的通道数要保持一致 
在while True 循环时 img 不能变动图片通道数
'''
class Track:
    def __init__(self,path):

        self.img=cv2.imread(path)
        self.ims=self.img
        
        self.hsv=cv2.cvtColor(self.img,cv2.COLOR_BGR2HSV)

        self.lower=np.array([20,30,50])
        self.uper=np.array([100,120,200])

    def fun(self,n):
        
        h_min=cv2.getTrackbarPos('h_min','img')
        h_max=cv2.getTrackbarPos('h_max','img')
        s_min=cv2.getTrackbarPos('s_min','img')
        s_max=cv2.getTrackbarPos('s_max','img')
        v_min=cv2.getTrackbarPos('v_min','img')
        v_max=cv2.getTrackbarPos('v_max','img')
        
        self.lower=np.array([h_min,s_min,v_min])
        self.uper=np.array([h_max,s_max,v_max])

        mask=cv2.inRange(self.hsv,self.lower,self.uper)

        # self.ims=cv2.merge([mask,mask,mask])
        self.ims=cv2.bitwise_and(self.img,self.img,mask=mask)


    def run(self):
        '''
        跟鼠标事件一样 1创建窗口  2设置滑动条  3显示图片
        '''
        cv2.namedWindow('img',cv2.WINDOW_NORMAL)

        cv2.createTrackbar('h_min','img',0,255,self.fun) # 第三个参数 设置0
        cv2.createTrackbar('h_max','img',0,255,self.fun)
        cv2.createTrackbar('s_min','img',0,255,self.fun)
        cv2.createTrackbar('s_max','img',0,255,self.fun)
        cv2.createTrackbar('v_min','img',0,255,self.fun)
        cv2.createTrackbar('v_max','img',0,255,self.fun)

        self.show()
        
    def show(self):
        while True:
            cv2.imshow('img',self.ims)
            key=cv2.waitKey(10)
            if key==ord('x'):
                break
        cv2.destroyWindow('img')

    # def show(self,img):
    #     while True:
    #         cv2.imshow('img',img)
    #         key=cv2.waitKey(10)
    #         if key==ord('x'):
    #             break
    #     cv2.destroyWindow('img')
    

if __name__ == '__main__':
    path=r'./imgs/sea.jpg'
    t=Track(path)
    t.run()
