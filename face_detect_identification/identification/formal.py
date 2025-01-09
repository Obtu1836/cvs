import os
import cv2 
import numpy as np 

class Seiko:
    def __init__(self,path):

        self.detect=cv2.CascadeClassifier(path)
        
    def make_dataset(self,path,imgsize=(200,200)):

        names=[]
        label=0
        train_img,train_label=[],[]
        for dirs,subdirs,files in os.walk(path):
            for subdir in subdirs:
                names.append(subdir)
                subject_path=os.path.join(dirs,subdir)
                for files in os.listdir(subject_path):
                    img=cv2.imread(os.path.join(subject_path,files),cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    img=cv2.resize(img,imgsize)
                    train_img.append(img)
                    train_label.append(label)
                label+=1

        train_data=np.asarray(train_img,dtype=np.uint8)
        train_label=np.asarray(train_label,dtype=np.int32)

        return names,train_data,train_label
    

    def trainx(self,path):

        # self.identfi=cv2.face.EigenFaceRecognizer_create()
        self.identfi=cv2.face.FisherFaceRecognizer_create()
        self.identfi=cv2.face.LBPHFaceRecognizer_create()
        self.names,self.train_data,self.train_label=self.make_dataset(path)
        self.identfi.train(self.train_data,self.train_label)

    def predict(self,img):

        fs=self.detect.detectMultiScale(img)

        for (x,y,w,h) in fs:
            roi=img[y:y+h,x:x+w]
            gray_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            gray_roi=cv2.resize(gray_roi,(200,200))
            label,confidence=self.identfi.predict(gray_roi)

            text=f'{self.names[label]},{confidence:.3f}'

            cv2.putText(img,text,(x,y+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
        
        return img


    
        
if __name__ == '__main__':
    
    model_path=r'face_detect_identification/model/haarcascade_frontalface_default.xml'
    img_path=r'face_detect_identification/imgs'
    s=Seiko(model_path)
    s.trainx(img_path)


    dirs=r'/Users/yan/Books/OPENCV-4/data/at/jh'
    files=[os.path.join(dirs,var) for var in os.listdir(dirs)]
    for file in files :
        if 'pgm' in file:
            ims=cv2.imread(file)
            ims=s.predict(ims)

            cv2.imshow('res',ims)
            key=cv2.waitKey(100)
            if key==ord('x'):
                break



