import cv2 
import numpy as np 

def rotate(img,angle):

    '''
    旋转图片 也可以看作是反方向旋转坐标轴
    先将坐标轴平移到旋转中心 将坐标轴旋转-degree 等价于图片旋转degree
    根据旋转后的图形计算新的宽高
    根据新的宽高计算出 新的中心 再将原点坐标 按新的中心反向平移 
    '''

    h,w=img.shape[:2]
    cx,cy=w//2,h//2  # 这一步需要整除 是因为getRotationMatrix2D((cx,cy)) cx,cy输入需要整数

    m=cv2.getRotationMatrix2D((cx,cy),-angle,1)
    cos,sin=np.abs(m[:,0]) # 计算出 cos(theta) 和sin(theta)的绝对值

    nw=int(w*cos+h*sin) # 计算出旋转后图片的宽和高 
    nh=int(w*sin+h*cos)
    
    m[0,2]+=nw/2-cx   #计算 新图和后图 旋转中心的偏移量   也可以用(nw//2-cx) | (nh//2-cy)效果差不多
    m[1,2]+=nh/2-cy

    ims=cv2.warpAffine(img,m,(nw,nh),None,2,0,(255,255,255))

    return ims

if __name__ == '__main__':
    
    path=r'imgs/dat.png'

    img=cv2.imread(path)

    flag=True
    i=0
    while flag:
        ims=rotate(img,i)
        cv2.imshow('ims',ims)
        key=cv2.waitKey(10)
        if key==ord('x'):
            break
        i+=1
        if i==360:
            i=0



