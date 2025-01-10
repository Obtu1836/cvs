import cv2
import argparse
import numpy as np

'''
yolov8的输出 (84,8400) 所以需要转置一下
84 前4个数字 代表 cx,cy,w,h 后80是80个类别 置信度融合到80个类别里 
'''


class Seiko:
    def __init__(self, model_path, names_path, conf_thres, iou_thres, img_path, shape):

        self.net = cv2.dnn.readNet(model_path)
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
        outs = outs.T

        box,name=self.predict(outs)
        ori_box = self.anti_letter(box, diff, radio)
        self.draw_rect(ori_box, name)


    def letterbox(self, value=None):

        shape = np.array(self.shape)
        h, w = self.img.shape[:2]
        ori_shape = np.array([w, h])
        radio = min(shape/ori_shape)
        if value is None:
            value = self.img.mean((0, 1)).astype(np.uint8).tolist()
        mid_shape = (ori_shape*radio).astype(np.int32)
        diff = shape-mid_shape
        ims = cv2.resize(self.img, mid_shape, interpolation=cv2.INTER_AREA)
        d1 = diff//2
        d2 = diff-d1
        ims = cv2.copyMakeBorder(ims, d1[1], d2[1], d1[0], d2[0],
                                 cv2.BORDER_CONSTANT, value=value)

        return ims, d1, radio

    def predict(self, outs):

        boxs, names,confidents= [], [],[]

        for var in outs:
            boxs.append(var[:4])
            scores=var[4:]
            ind = np.argmax(scores)
            confidents.append(scores[ind])
            names.append(self.names[ind])

        boxs = np.array(boxs).astype(np.int32)
        boxs = self.cxcywh2xywh(boxs)
        names = np.array(names)
        confidents=np.array(confidents)

        ix = cv2.dnn.NMSBoxes(
            boxs, confidents, self.conf_thres, self.iou_thres)
        
        box = boxs[ix]
        name = names[ix]

        return box,name

    def cxcywh2xywh(self, boxs):

        boxs[:, 0] -= boxs[:, 2]//2
        boxs[:, 1] -= boxs[:, 3]//2

        return boxs
    
    def anti_letter(self, box, diff, radio):

        box[:, :2] -= diff
        box = (box/radio).astype(np.int32)

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='yolo/models/yolov8s.onnx')
    parser.add_argument('--names_path', type=str,
                        default='yolo/models/coco.names')
    parser.add_argument('--conf-thres', type=float, default=0.2)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--img_path', type=str,
                        default='yolo/models/demo1.jpg')
    parser.add_argument('--shape', type=int, default=[640], nargs='+')

    opt = parser.parse_args()
    opt.shape *= 2 if len(opt.shape) == 1 else 1

    s = Seiko(**vars(opt))
    s.run()
