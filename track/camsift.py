import cv2 
import numpy as np 

'''
总体思路和meanshift一样 只不过 camshift 第一个返回值是 点

使用cv2.boxpoint将这些点 转化为顶点 在画出图形
'''

class Seiko:
    def __init__(self):

        self.cap=cv2.VideoCapture(0)
        self.eps=(cv2.TermCriteria_COUNT|cv2.TERM_CRITERIA_EPS,10,1)
    
    def run(self):

        for i in range(30):
            _,frame=self.cap.read()
            if i==29:
                hist,window=self.roi_hist(frame)
        
        while True:
            _,img=self.cap.read()
            hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            back=cv2.calcBackProject([hsv],[0],hist,[0,180],1)

            rota_rect,window=cv2.CamShift(back,window,self.eps)
            box=cv2.boxPoints(rota_rect).astype(np.int32)
            cv2.polylines(img,[box],True,(0,255,2),2)

            cv2.imshow('img',img)
            key=cv2.waitKey(1)
            if key==ord('x'):
                break
            
        self.cap.release()
        cv2.destroyAllWindows()

    def roi_hist(self,frame):
        x,y,w,h=cv2.selectROI('roi',frame)
        window=(x,y,w,h)
        roi=frame[y:y+h,x:x+w]
        hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        hist=cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
        cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)

        return hist,window
    

if __name__ == '__main__':
    
    s=Seiko()
    s.run()
