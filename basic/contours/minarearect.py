import cv2 
import numpy as np 

path=r'./imgs/hammer.jpg'
img=cv2.imread(path)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh=cv2.adaptiveThreshold(gray,255,cv2.THRESH_BINARY,1,5,10)
cons,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

cv2.drawContours(img,cons,-1,(0,0,255),2)
box=cv2.minAreaRect(cons[0])# 返回 （（x,y）,(w,h),angle）
'''
旋转角度θ是水平轴（x轴）逆时针旋转,直到碰到矩形的第一条边停住,此时该边与水平轴的夹角。
并且这个边的定义为width,另一条边定义为height。
也就是说,在这里,width与height不是按照长短来定义的
'''
box=cv2.boxPoints(box).astype(np.int32) # 返回矩形的4个顶点坐标

cv2.drawContours(img,[box],-1,(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)

