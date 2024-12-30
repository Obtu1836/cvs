import cv2 
import numpy as np 
img=cv2.imread(r'imgs/lun.jpeg')

h,w=img.shape[:2]
m=np.float32([[1,0,100], # 图像向右平移 100
              [0,1,50]]) # 图像向下平移 50
res=cv2.warpAffine(img,m,(w+200,h+100)) # 第三个参数控制最后图像的大小

cv2.imshow('res',res)
cv2.waitKey(0)