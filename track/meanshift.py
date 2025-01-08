import cv2 
import numpy as np 

'''
打开摄像头 自动等待30帧 以保持摄像头稳定状态
通过roi选取第30帧的感兴趣部分 将这部分单独取出来 转hsv 通过calhist函数 计算
hist(0通道的) 在标准化

通过calcBackProject 直方图反投影 并将该函数的结果 送入
meanshift(密度聚类) 得到新的位置

calcBackProject函数和calcHist函数的各个参数传入时  以列表的形式
'''
class Seiko:
    def __init__(self):

        self.cap=cv2.VideoCapture(0)
        self.eps=(cv2.TermCriteria_COUNT|cv2.TermCriteria_EPS,10,1)
    
    def run(self):

        for i in range(30):
            _,frame=self.cap.read()
            if i==29:
                hist,window=self.roi_hist(frame)

        while True:
             
            _,img=self.cap.read()
            hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            back=cv2.calcBackProject([hsv],[0],hist,[0,180],1)
            num_iter,window=cv2.meanShift(back,window,self.eps)#window和window是相互迭代的
            '''
            meanshift返回值 迭代次数和新的窗口 如果num_iter<self.eps里的count
            则 结果一定是收敛的 
            '''

            x,y,w,h=window
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.imshow('img',img)
            key=cv2.waitKey(1)
            if key==ord('x'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


    def roi_hist(self,frame):

        (x,y,w,h)=cv2.selectROI('roi',frame)
        window=(x,y,w,h)
        roi=frame[y:y+h,x:x+w]
        hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        hist=cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
        cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)

        return hist,window

if __name__ == '__main__':
    
    s=Seiko()
    s.run()
