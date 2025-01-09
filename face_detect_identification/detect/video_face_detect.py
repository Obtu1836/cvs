import cv2 
import numpy as np 

class Seiko:
    def __init__(self,model_path):

        self.cap=cv2.VideoCapture(0)
        self.casde=cv2.CascadeClassifier(model_path)
    
    def run(self):

        while True:
            _,img=self.cap.read()
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            fs=self.casde.detectMultiScale(gray,1.12,5,0,minSize=(50,50))
            
            for (x,y,w,h) in fs:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.imshow('img',img)
            key=cv2.waitKey(1)

            if key==ord('x'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    model_path=r'face_detect_identification/model/haarcascade_frontalface_default.xml'
    s=Seiko(model_path)

    s.run()
            

        