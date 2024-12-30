import cv2
import numpy as np

'''
1 提取轮廓 将轮廓内的坐标点(xs,ys)--->xys记录 拟合出圆心位置(cx,cy)
2 旋转 首先平移坐标系 将坐标系平移到圆心的位置(xys-center) 然后根据M计算出旋转以后的位置坐标 再将坐标系反向
移动到(0,0) new_points+center  这样新的坐标就是图像坐标系下的坐标 在转化为整数

3 在上述转化整数过程中 因为小数的存在 必然会丢失一些像素信息 所以需要插值 采用的双线性插值方法

4插值的思路:
  1 首先确定出哪些点被忽略了 xys是全部坐标点 new_points是转化后的点 也就意味着 找出xys与new_points的
  差集 就为丢失的像素点 miss_poins
  2 将miss_point经rotationx函数在反向旋转回原位 新的坐标点同样为小数 对这些小数坐标进行线性插值 得到
  每个小数坐标点的像素值
  3 这些小数坐标对应的像素值 就为miss_point对应的像素值 使用mask[miss_point[:,1],miss_point[:,0]]=value
  即可
'''


class Seiko:
    def __init__(self, path, angle):

        self.img = cv2.imread(path)
        self.angle = angle

    def extract(self):  # 提取轮廓 拟合出圆心 这个圆心将作为旋转的初始坐标系原点

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = contours[0]

        (cx, cy), r = cv2.minEnclosingCircle(contours)

        return int(cx), int(cy), int(r), thresh

    def run(self):

        cx, cy, r, bin_img = self.extract()
        ys, xs = np.where(bin_img == 255)  # 提取轮廓内的坐标点
        xys = np.c_[xs, ys]

        angle = 0
        while True:

            mask = 255-np.zeros(self.img.shape, np.uint8)  # 建一个新的白纸
            new_points = self.rotationx(
                xys, cx, cy, angle).astype(np.int32)  # 计算原轮廓内坐标点
            # 经旋转一点角度以后新的坐标点 并转化为整数

            '''
            计算坐标点差集 下面这种方法速度快,准确
            # '''
            points_all = xys.view([('', xys.dtype)] * xys.shape[1])
            points_new = new_points.view(
                [('', new_points.dtype)] * new_points.shape[1])
            miss_point = np.setdiff1d(points_all, points_new).view(
                xys.dtype).reshape(-1, 2)

            '下面这段是常规方法 比较慢'
            # ind=np.where((xys[:,None]==new_points).all(axis=2))[0]
            # non_ind=np.setdiff1d(np.arange(len(xys)),ind)
            # miss_point=xys[non_ind]

            miss_point_ori = self.rotationx(
                miss_point, cx, cy, -angle)  # 将找到的差集在反向旋转到原图
            value = self.linear_interpolation(miss_point_ori)  # 对这些坐标点双线性插值

            mask[new_points[:, 1], new_points[:, 0]
                 ] = self.img[ys, xs]  # 将整数点像素值映射
            mask[miss_point[:, 1], miss_point[:, 0]] = value  # 将缺失点的像素值映射

            cv2.imshow('mask', mask)
            key = cv2.waitKey(1)
            if key == ord('x'):
                break
            angle += self.angle
            if angle > 360:
                angle = 0

    def rotationx(self, xys, cx, cy, angle):
        '''
        旋转函数 首先平移圆心 再进行旋转 再平移回圆心
        '''

        angle = angle*np.pi/180
        xys = xys-[cx, cy]  # 不能-=这种增量写法 会改变xys的数值 从而影响别的函数使用xys
        m = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])

        new_point = (xys.dot(m))
        new_point = new_point+[cx, cy]

        return new_point

    def linear_interpolation(self, empty):
        '''
        向量化的双线性插值函数 速度超级快
        '''

        x, y = np.split(empty, [1], axis=1)
        xf = np.floor(x)
        yf = np.floor(y)

        cornerx = np.array([0, 0, 1, 1])  # 分别是左上 左下  右下 右上的边界阈值
        cornery = np.array([0, 1, 1, 0])

        idx = (xf+cornerx).astype(np.int32)  # 小数坐标点的4个临近的整数点
        idy = (yf+cornery).astype(np.int32)

        points = self.img[idy, idx]  # 获取4个整数点的像素值

        radio_x = 1-(x-xf)  # 确定左方权重
        radio_y = 1-(y-yf)  # 确定上方权重

        diffx = np.abs(radio_x-cornerx[None, :])  # 右方权重
        diffy = np.abs(radio_y-cornery[None, :])  # 下方权重
        raido = (diffx*diffy)

        values = (points*raido[:, :, None]).sum(axis=1)
        return values


if __name__ == '__main__':

    path = r'imgs\lun2.png'
    s = Seiko(path, 12)
    s.run()
