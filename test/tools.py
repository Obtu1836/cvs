import numpy as np 
import cv2 

def seiko(left):
    '''
    裁剪
    '''
    img=cv2.imread(path)
    x,y,w,h=cv2.selectROI('rect',img)
    # cv2.rectangle(img,(x,y),(x+w,y+h),(0.0,255),2)
    ims=img[y:h+y,x:x+w]

    cv2.imwrite(f'./imgs/{left}.png',ims)


def seiko1(path):
    '''
    放缩
    '''
    img=cv2.imread(path)
    ims=cv2.resize(img,None,fx=0.3,fy=0.3,interpolation=cv2.INTER_AREA)

    cv2.imwrite(r'./imgs/lun1.png',ims)

def sony(path):
    img=cv2.imread(path)
    h,w=img.shape[:2]

    mask=np.zeros((600,800,3))
    mask[250:250+h,100:100+w]=img
    mask=mask.astype(np.uint8)
    cv2.imshow('mask',mask)
    key=cv2.waitKey(0)
    if key==ord('x'):
        cv2.imwrite(r'imgs/lun3.png',mask)

if __name__ == '__main__':
    
    path=r'imgs\lun1.png'

    sony(path)
    
