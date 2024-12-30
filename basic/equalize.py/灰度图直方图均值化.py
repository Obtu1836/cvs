import cv2 
import numpy as np 

def single(path):
    '''单通道'''
    gray=cv2.imread(path,0)

    dst=cv2.equalizeHist(gray)
    new=np.c_[gray,dst]
    cv2.imshow('img',new)
    cv2.waitKey(0)

def mul_chanle(path):

    '''
    多通道的直方图均衡 是对每个通道分别均衡 然后再组合一下
    '''
    img=cv2.imread(path)

    r,g,b=cv2.split(img) # 分离通道
    rh=cv2.equalizeHist(r)
    gh=cv2.equalizeHist(g)
    bh=cv2.equalizeHist(b)

    result=cv2.merge((rh,gh,bh))# 合并通道

    new=np.concatenate([img,result],axis=1)
    cv2.imshow('new',new)
    cv2.waitKey(0)

if __name__ == '__main__':
    path=r'./imgs/sea.jpg'
    # single(path)
    mul_chanle(path)
