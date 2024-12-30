import cv2
import numpy as np

'''
通过前景照片 制作两个掩膜 fg_mask 和fg_inv_mask

通过前景照片和fg_mask bitwise_and  得到前景v
通过鼠标操作 得到roi
通过 背景照片和fg_inv_mask bitwise 得到背景k

最后 cv2.add(k,v)
'''

class Seiko:
    def __init__(self, path1, path2):

        self.fgimg = cv2.imread(path1)
        self.bgimg = cv2.imread(path2)
        self.points = None

    def run(self):

        fg_mask = self.getmask(self.fgimg) # 实例 255 其余0
        fg_inv_mask = cv2.bitwise_not(fg_mask)# 非实例部分255 其余0

        bg_roi, (left, right, up, down) = self.make_roi()#得到roi

        k = cv2.bitwise_and(bg_roi, bg_roi, mask=fg_inv_mask)
        v = cv2.bitwise_and(self.fgimg, self.fgimg, mask=fg_mask)

        res = cv2.add(k, v)
        self.bgimg[up:down, left:right] = res

        self.show(self.bgimg)

    def getmask(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return thresh

    def fun(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points = (x, y)

    def make_roi(self):

        cv2.namedWindow('roi', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('roi', self.fun)
        while True:
            cv2.imshow('roi', self.bgimg)
            key = cv2.waitKey(1)
            if key == ord('x'):
                break
        cv2.destroyWindow('roi')
        
        x, y = self.points
        fh, fw = self.fgimg.shape[:2]
        bh, bw = self.bgimg.shape[:2]
        left, right = x-fw//2, x+fw-fw//2
        up, down = y-fh//2, y+fh-fh//2

        bo = np.array([left > 0, up > 0, right < bw, down < bh])
        word = np.array(['l', 'u', 'r', 'd'])

        assert bo.all(), f'{word[np.where(~bo)[0]]} 超界'

        roi = self.bgimg[up:down, left:right]
        return roi, (left, right, up, down)

    def show(self, img, name='img'):
        
        cv2.imshow(name, img)
        cv2.waitKey(0)


if __name__ == '__main__':
    path1 = r'imgs/cv.png'
    path2 = r'imgs/china.jpeg'

    seiko = Seiko(path1, path2)
    seiko.run()
