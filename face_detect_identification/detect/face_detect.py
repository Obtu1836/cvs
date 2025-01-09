import cv2 
import numpy as np 

def sony(model_path,img_path):

    face=cv2.CascadeClassifier(model_path)

    img=cv2.imread(img_path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    fs=face.detectMultiScale(gray,1.08,5,maxSize=(50,50))#灰度图 彩色图都可以
    print(fs)

    for (x,y,w,h) in fs:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imshow('img',img)
    cv2.waitKey(0)

if __name__ == '__main__':
    
    model_path=r'face_detect_identification/model/haarcascade_frontalface_default.xml'
    img_path=r'/Users/yan/Code/cvs/imgs/woodcutters.jpg'
    sony(model_path,img_path)

