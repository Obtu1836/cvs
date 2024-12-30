import cv2 
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt

path=r'imgs\sea.jpg'
pic=Image.open(path)

data=np.asarray(pic)

img=cv2.cvtColor(data,cv2.COLOR_RGB2BGR)

fig=plt.figure(figsize=(2,1.5),dpi=300)
plt.imshow(data)
plt.axis('off')
plt.show()
