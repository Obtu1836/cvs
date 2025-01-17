import cv2
import numpy as np
import logging
import argparse

np.set_printoptions(precision=4, suppress=True)

'''
总体的思路 
1 brisk提取关键点 然后匹配 等常规操作
2 确定最终的形状以及忽视照片的相对位置关系 思路是 通过关键点 确定出 M
  然后通过M 把第一张图全图投影到第二张图 计算顶点坐标 如果xmin<0&ymin<0 调换一下
  img1<->img2,src_point,dst_point<->dst_point,src-point 重新计算 M
3 根据第二步计算出M 将img1 warp 得到 reslut 这个中间状态,然后比较img_img和
  img2的面积大小 如果img2.size<reslut.size 调换一下顺序 img2<->reslut,
  这样做的原因总能保持 img2这个变量代表的是面积是相对大的 方便后续操作
4 因为img2是大的 所以reslut这个就是小的 进入image_fushion函数 将reslut通过函数
  cv2.copyMakeBorder扩展到和img2相同大小

  a np.where(big==0,big,small)这个方法 可以找到2个图像交集(重叠的部分)
  b 分别二值化 small,coincidence,big这三个图像 通过二值化后的图像相加减
        得到left,coincidence,right轮廓 
    xs,ys=np.where(coincidence==255) 得到重叠部分的坐标

    然后根据cv2.pointPolygonTest这个函数 计算这些图像的坐标到left和right
    轮廓的距离 根据这些距离加权 使图像加权融合

    findHomography和getPerspectiveTransform 都是通过点来确定透视变换矩阵m 
    其中findHomography是通过一系列的点 getPerspectiveTransform可以通过4个点 
    warpwarpPerspective函数 是通过m 变换对应的图像
    perspectiveTransform函数 是通过m 变换对应坐标


'''
class Seiko:
    def __init__(self, path1, path2, draw_match, draw_contour, kp_length):

        self.img1 = cv2.imread(path1)
        self.img2 = cv2.imread(path2)
        self.draw_mt = draw_match
        self.draw_cr = draw_contour
        self.kp_length = kp_length

    def kp_desc(self):
        brisk = cv2.BRISK.create()
        kp1, desc1 = brisk.detectAndCompute(self.img1, None)
        desc1 = desc1.astype(np.float32)
        kp2, desc2 = brisk.detectAndCompute(self.img2, None)
        desc2 = desc2.astype(np.float32)

        index = dict(algorithm=1, trees=5)
        search = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index, search)
        matches = flann.knnMatch(desc1, desc2, k=2)

        return matches, kp1, kp2

    def good_mask(self, mat):

        mask = [(0, 0) for i in range(len(mat))]
        good = []
        for i, (m, n) in enumerate(mat):
            if m.distance < 0.7*n.distance:
                mask[i] = (1, 0)
                good.append(m)
        return good, mask

    def run(self):

        matches, kp1, kp2 = self.kp_desc()
        good, mask = self.good_mask(matches)
        assert len(good) > self.kp_length, '匹配点不足'

        if self.draw_mt:
            ps = cv2.drawMatchesKnn(self.img1, kp1, self.img2, kp2, matches, None, None, (255, 0, 0),
                                    mask, cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            self.show(ps, 'matches')

        
        src_point = np.float32([kp1[m.queryIdx].pt for m in good])
        dst_point = np.float32([kp2[m.trainIdx].pt for m in good])

        xmin, xmax, ymin, ymax, m = self.cal_m(src_point, dst_point)

        if xmin < 0 and ymin < 0:  # 如果 xmin<0 and ymin<0 重新投影一次
            logging.info('img1<-->img2')
            self.img1, self.img2 = self.img2, self.img1  # 调换图像
            src_point, dst_point = dst_point, src_point  # 调换关键点
            xmin, xmax, ymin, ymax, m = self.cal_m(src_point, dst_point)

        result = cv2.warpPerspective(self.img1, m, (xmax, ymax))  # 中间状态

        if self.img2.size < result.size:  # 比较大小
            logging.info('img<-->result')
            self.img2, result = result, self.img2  # 保持img2最大
        res = self.image_fushion(self.img2, result)

        self.show(res, 'res')

    def cal_m(self, src, dst):

        m, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5)
        h, w = self.img1.shape[:2]
        corner = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        map_corner = cv2.perspectiveTransform(
            corner, m).astype(np.int32).reshape(-1, 2)
        '''投影img1 确定xmin,ymin xmax,ymax'''
        xmin, xmax = map_corner[:, 0].min(), map_corner[:, 0].max()
        ymin, ymax = map_corner[:, 1].min(), map_corner[:, 1].max()

        return xmin, xmax, ymin, ymax, m

    def image_fushion(self, big, small):

        res = big.copy()
        big_mask = self.binarization(big)
        bh, bw = big.shape[:2]
        sh, sw = small.shape[:2]

        res[:sh, :sw] = np.maximum(res[:sh, :sw], small)

        small_x = cv2.copyMakeBorder( #将small图像 扩展成和big大小一样的图片
            small, 0, bh-sh, 0, bw-sw, cv2.BORDER_CONSTANT, None, (0, 0, 0))
        small_mask = self.binarization(small_x)

        coincidence = self.binarization(np.where(big == 0, big, small_x))
        con_contours = self.get_contours(coincidence)
        '''重叠部分的轮廓'''

        left_mask = cv2.subtract(small_mask, coincidence)
        left_contours = self.get_contours(left_mask)
        '''small-重叠=非重叠部分的轮廓'''
        right_mask = cv2.subtract(big_mask, coincidence)
        right_contours = self.get_contours(right_mask)
        '''big-重叠=非重叠部分的轮廓'''

        # 计算距离 采用apply_along_axis bing没有比for循环加快运行速度！
        pointy, pointx = np.where(coincidence == 255)
        # cv2.pointPolygonTest这个函数点的坐标需要float
        points = np.c_[pointx, pointy].astype(np.float32)
        dis = np.apply_along_axis(
            self.cal_dis, 1, points, left_contours, right_contours)
        dis=dis/dis.sum(axis=1)[:,None]
        dis[:,[0,1]]=dis[:,[1,0]]

        if self.draw_cr:

            sk = res.copy()
            cv2.drawContours(sk, [left_contours], 0, (0, 255, 0), 2)
            cv2.drawContours(sk, [con_contours], 0, (255, 0, 0), 2)
            cv2.drawContours(sk, [right_contours], 0, (0, 0, 255), 2)
            self.show(sk, 'all_contours')

        s_value = small_x[pointy, pointx]
        b_value = big[pointy, pointx]
        res[pointy, pointx] = np.clip(
            s_value*dis[:, 0][:, None]+b_value*dis[:, 1][:, None], 0, 255).astype(np.int32)

        return res

    def get_contours(self, img):

        cons, _ = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        try:
            assert len(cons) >= 1, 'mask failed'

        except AssertionError as e:
            logging.info(e)
            self.show(img, 'error! contours not found')
        cons = sorted(cons, key=cv2.contourArea)[::-1][0]
        return cons

    def binarization(self, img):  # 二值化  threshold 参数0 很必要

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        return thresh

    def cal_dis(self, arr, scons, bcons):  # 计算距离 arr的type=float

        sdis = -cv2.pointPolygonTest(scons, arr, True)
        bdis = -cv2.pointPolygonTest(bcons, arr, True)

        return sdis, bdis

    def show(self, img, name='result'):
        cv2.imshow(name, img)
        cv2.waitKey(0)


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('--path1', type=str,
                       default=r'imgs\left.png')
    parse.add_argument('--path2', type=str,
                       default=r"imgs\right.png")
    parse.add_argument('--draw_match', action='store_true', default=False)
    parse.add_argument('--draw_contour', action='store_true', default=False)
    parse.add_argument('--kp_length', type=int, default=4)
    opt = parse.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d  %H:%M:%S ')

    s = Seiko(**vars(opt))
    s.run()
