import cv2
import argparse
import numpy as np
import logging

'''
一 opencv 读取onnx网络可以使用 dnn.readnetfromonnx 也可以直接用readnet
读取网络3步走
    1 net=cv2.dnn.readnet(path) 加载模型
    2 layername=net.getlayernames() 读取网络结构 获取每个层的名字
    3 output=[layername[i-1] for i in net.getunconnectedoutlayers()]
     获取输出层

二 yolov5的输出 (25200,85)
85 前4个数字 代表 cx,cy,w,h 第5个是置信度confident 后80是80个类别

三 letterbox函数  防止输入的图片经过 经过resize后产生畸变 实现的方法是
 以短边为缩放基础 等比例缩放图片至长边等于要求的长度 此时短边肯定不到要求的长度
 所以利用cv2.copymakeboard函数将短边补齐至要求的长度
'''


class Seiko:
    def __init__(self, model_path, names_path, conf_thres, 
                 iou_thres, img_path, shape):

        self.net = cv2.dnn.readNetFromONNX(model_path)
        layername = self.net.getLayerNames()
        self.outputs = [layername[i-1]
                        for i in self.net.getUnconnectedOutLayers()]

        with open(names_path, 'r') as r:
            self.names = [line.strip() for line in r.readlines()]

        self.shape = shape
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img = cv2.imread(img_path)

    def run(self):

        ims, diff, radio = self.letterbox()
        blob = cv2.dnn.blobFromImage(ims, 1/255, self.shape, [0, 0, 0],
                                     True, False)
        self.net.setInput(blob)
        outs = self.net.forward(self.outputs)[0][0]

        logging.info(f'网络输出形状：{outs.shape}')

        box, names = self.predict(outs)

        ori_box = self.anti_letter(box, diff, radio)

        self.draw_rect(ori_box, names)

    def letterbox(self,value=None):

        shape = np.array(self.shape)  # 获取要输入的形状
        h, w = self.img.shape[:2]
        ori_shape = np.array([w, h])  # 图片本身的形状 w在前 h在后 后续计算方便
        radio = min(shape/ori_shape)  # 以短边 确定图片的缩放比例

        if value is None:  # 以通道的均值作为基准颜色 补齐
            value = self.img.mean((0, 1)).astype(np.uint8).tolist()

        mid_shape = (ori_shape*radio).astype(np.int32)  # 根据缩放比例 缩放图片
        ims = cv2.resize(self.img, mid_shape, interpolation=cv2.INTER_AREA)
        diff = shape-mid_shape  # 看看短边与 要输入的形状差多少
        d1 = diff//2  # 分别在两边进行扩展 假如diff=11 高度 那么 上边补5 下边补6
        d2 = diff-d1
        ims = cv2.copyMakeBorder(ims, d1[1], d2[1], d1[0], d2[0],
                                 cv2.BORDER_CONSTANT, value=value)

        return ims, d1, radio  # 返回图片 d1 radio d1是上或者左补齐的像素数量

    def predict(self, outs):

        boxs, confidents, names = [], [], []
        for var in outs:
            confident = var[4]  # 置信度
            if confident < self.conf_thres:  # 初步过滤一下置信度低的
                continue
            confidents.append(confident)
            scores = var[5:]  # 类别概率
            ind = np.argmax(scores)
            names.append(self.names[ind])
            boxs.append(var[:4])  # cx,cy,w,h

        boxs = np.array(boxs).astype(np.int32)
        boxs = self.cxcywh2xywh(boxs)  # cx,cy,w,h--->x,y,w,h
        names = np.array(names)
        confidents = np.array(confidents)

        ix = cv2.dnn.NMSBoxes(
            boxs, confidents, self.conf_thres, self.iou_thres)
        ''' opencv的非极大值抑制 输入的box 是(x,y,w,h)格式'''
        box = boxs[ix]
        name = names[ix]

        return box, name

    def cxcywh2xywh(self, boxs):

        boxs[:, 0] -= boxs[:, 2]//2
        boxs[:, 1] -= boxs[:, 3]//2

        return boxs

    def anti_letter(self, box, d1, radio):
        '''
        反letterbox  d1 是左或者上的补齐的像素数量 减去d1 也就意味着坐标系原点
        从有补齐的图片原点平移到 没有补齐的图像的原点
        box-=d1 可以理解为平移 
        '''
        box[:, :2] -= d1  # 先平移
        box = (box/radio).astype(np.int32)  # 在除以放缩系数 还原
        return box

    def draw_rect(self, box, name):

        for i, var in enumerate(box):
            x, y, w, h = var
            cv2.rectangle(self.img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(self.img, name[i], (x, y-20), cv2.FONT_HERSHEY_COMPLEX,
                        0.6, (0, 0, 255), 1)
        self.show(self.img)

    def show(self, img, name='img'):
        cv2.imshow(name, img)
        cv2.waitKey(0)


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d  %H:%M:%S ')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='yolo/models/yolov5s.onnx')
    parser.add_argument('--names_path', type=str,
                        default='yolo/models/coco.names')
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--img_path', type=str,
                        default='yolo/models/demo1.jpg')
    parser.add_argument('--shape', type=int, default=[640], nargs='+')
    opt = parser.parse_args()
    opt.shape *= 2 if len(opt.shape) == 1 else 1

    s = Seiko(**vars(opt))
    s.run()
