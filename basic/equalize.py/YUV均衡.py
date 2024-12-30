import cv2 
import numpy as np 

def seiko(path):
    img=cv2.imread(path,1)
    imgyuv=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)#转化色彩空间

    channel=list(cv2.split(imgyuv))#分割通道
    channel[0]=cv2.equalizeHist(channel[0])# 只对第一个通道均衡
    chanle=cv2.merge(channel) #然后再合并通道
    res=cv2.cvtColor(chanle,cv2.COLOR_YCrCb2BGR)# 转回BGR
    new=np.concatenate([img,res],axis=1)
    cv2.imshow('yuv',new)
    cv2.waitKey(0)

if __name__ == '__main__':
    path=r'./imgs/sea.jpg'
    seiko(path)