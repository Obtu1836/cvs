import cv2 
import numpy as np 
import matplotlib.pyplot as plt

def cal_gray(img):
    '''灰度图 计算直方图  常规方法
    '''
    row,col=img.shape
    xy=np.zeros([256],np.uint64)
    for r in range(row):
        for c in range(col):
            xy[img[[r],[c]]]+=1
    return xy

def main(path):

    img=cv2.imread(path,0)
    xy=cal_gray(img)
    x_range=range(256)

    plt.plot(x_range,xy,color='r')
    plt.show()

def cal_hist(path):
    '''
    多通道计算 就是分别对每个通道分别计算一次
    '''
    img=cv2.imread(path)
    color=('b','g','r')
    for i ,col in enumerate(color):
        hist=cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(hist,color=color[i])
    plt.show() 

if __name__ == '__main__':

    path=r'./imgs/sea.jpg'
    # main(path)
    cal_hist(path)
