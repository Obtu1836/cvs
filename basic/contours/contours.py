import cv2
import numpy as np 

path=r'./imgs/hammer.jpg'
img=cv2.imread(path)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

_,thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

cons,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
'''
cv2.findcontours 参数:
image:输入图像
mode:轮廓的模式。cv2.RETR_EXTERNAL只检测外轮廓
               cv2.RETR_LIST检测的轮廓不建立等级关系
               cv2.RETR_CCOMP建立两个等级的轮廓,上一层为外边界,内层为内孔的边界。如果内孔内还有连通物体，则这个物体的边界也在顶层；
               cv2.RETR_TREE建立一个等级树结构的轮廓。
method:轮廓的近似方法。
        cv2.CHAIN_APPROX_NOME存储所有的轮廓点，相邻的两个点的像素位置差不超过1；
        cv2.CHAIN_APPROX_SIMPLE压缩水平方向、垂直方向、对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需要4个点来保存轮廓信息

返回：      
    contours:返回的轮廓
    hierarchy:每条轮廓对应的属性                        
'''
cons=sorted(cons,key=cv2.contourArea)[::-1][:3]# 根据轮廓的面积排序 按出面积最大的前三个 

for c in cons:
    cv2.drawContours(img,[c],-1,(0,255,0),2) # 第二个参数为列表形式
    
cv2.imshow('thresh',img)
cv2.waitKey(0)