import cv2 
from PIL import Image
import matplotlib.pyplot as plt

path=r'imgs\sea.jpg'
cv_img=cv2.imread(path)

cv_img=cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)

pic=Image.fromarray(cv_img)
# pic.show()

plt.imshow(pic)
plt.show()