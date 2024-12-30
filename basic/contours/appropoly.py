import numpy as np 
import cv2 

img=cv2.imread(r'./imgs/hammer.jpg')
ims=cv2.pyrDown(img)

ret,thresh=cv2.threshold(cv2.cvtColor(ims,cv2.COLOR_BGR2GRAY),
                         127,255,cv2.THRESH_BINARY)
cons,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)

mask=np.zeros_like(img)
for c in cons:
    eps=0.01*cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,eps,True)

    hull=cv2.convexHull(c)

    cv2.drawContours(mask,[c],-1,(0,255,0),2)
    cv2.drawContours(mask,[approx],-1,(255,0,0),2)
    cv2.drawContours(mask,[hull],-1,(0,0,255),2)

cv2.imshow('hull',mask)
cv2.waitKey()

cv2.destroyAllWindows()