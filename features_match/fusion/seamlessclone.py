import cv2
import numpy as np

'''
图像融合 融合后的图像可能存在一定程度的变色 
mask参数 是前景图像上的制作roi掩膜 掩膜内255区域 将会参与融合

cv2.Mixed_clone效果比较好

'''


class Seiko:
    def __init__(self, path1, path2):

        self.fg = cv2.imread(path1)
        self.bg = cv2.imread(path2)
        self.bgc = self.bg.copy()
        self.point = None

    def run(self):

        self.make_window('center', self.bgc)
        x, y = self.point
        mask = self.make_roi()

        res = cv2.seamlessClone(self.fg, self.bg, mask, (x, y),
                                cv2.MIXED_CLONE)
        self.show(res)

    def make_roi(self):

        mask = np.zeros(self.fg.shape[:2])
        x, y, w, h = cv2.selectROI('roi', self.fg)
        cv2.destroyWindow('roi')
        mask[y:y+h, x:x+w] = 255
        mask = mask.astype(np.uint8)
        return mask

    def fun(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.bgc, (x, y), 1, (0, 255, 0), 2)
            self.point = (x, y)

    def make_window(self, name, img):
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(name, self.fun)
        while True:
            cv2.imshow(name, img)
            key = cv2.waitKey(1)
            if key == ord('x'):
                break
        cv2.destroyWindow(name)

    def show(self, img, name='res'):
        cv2.imshow(name, img)
        cv2.waitKey(0)


if __name__ == '__main__':
    path1 = r'imgs/cv.png'
    path2 = r'imgs/china.jpeg'

    s = Seiko(path1, path2)
    s.run()
