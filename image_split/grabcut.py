import cv2 
import numpy as np 

'''
先用rect 确定大概的前景背景 然后用mask 进行微调

run函数 运行主体  mask_bg 进行背景微调 mask_fg记性前景微调

ims图像 全程不会变化 作用2个 1用于在这张图上  选定微调区域 2 全程计算依据这个图

img图像 会根据微调进行变化 用于显示微调效果

'''
class Grabcut:
    def __init__(self,path):

        self.img=cv2.imread(path)
        '''
        设立ims 为了防止因img随时变化 起固定作用
        '''
        self.ims=self.img.copy()
        '''
        下面三个变量 为全局变量 执行cv2.grabcut函数后 就地修改
        '''
        self.bg=np.zeros((1,65),dtype=np.float64)
        self.fg=np.zeros((1,65),dtype=np.float64)
        self.mask=np.zeros(self.img.shape[:2],dtype=np.uint8)

    def run(self):

        (x,y,w,h)=cv2.selectROI('rect',self.ims)
        rect=(x,y,x+w,y+h)
        cv2.destroyWindow('rect')
        cv2.grabCut(self.ims,self.mask,rect,self.bg,self.fg,
                    iterCount=1,mode=cv2.GC_INIT_WITH_RECT)
        
        mask1=np.where((self.mask==0)|(self.mask==2),0,1)
        mask1=mask1.astype(np.uint8)

        self.img=self.ims*mask1[:,:,None]

        while True:
            cv2.imshow('ims',self.ims)
            cv2.imshow('img',self.img)# img 随着操作变化 可已看出微调效果 ims用于选像素
            key=cv2.waitKey(1)
            if key==ord('x'): # 键盘事件 按x 销毁所有窗口
                # cv2.imwrite('./new.png',self.img)
                cv2.destroyAllWindows()
                break
            elif key==ord('b'):# 键盘事件 按b 执行微调背景
                self.mask_bg()
            elif key==ord('f'): # 键盘事件 按f 执行微调前景
                self.mask_fg()

    def mask_bg(self):

        cv2.namedWindow('bg',cv2.WINDOW_AUTOSIZE)
        (x,y,w,h)=cv2.selectROI('bg',self.ims) # 用的ims
        self.mask[y:y+h,x:x+w]=cv2.GC_PR_BGD # 微调选用可能的背景
        cv2.destroyWindow('bg') 
        cv2.grabCut(self.ims,self.mask,None,self.bg,self.fg,iterCount=1,mode=cv2.GC_INIT_WITH_MASK)
        ' 函数内部变量mask2 仅用于当前计算 不能直接用self.mask赋值'
        mask2=np.where((self.mask==0)|(self.mask==2),0,1)#
        mask2=mask2.astype(np.uint8)
        self.img=self.ims*mask2[:,:,None] # img变化  

    def mask_fg(self):

        cv2.namedWindow('fg',cv2.WINDOW_AUTOSIZE)
        (x,y,w,h)=cv2.selectROI('fg',self.ims)
        self.mask[y:y+h,x:x+w]=cv2.GC_PR_FGD
        cv2.destroyWindow('fg')
        cv2.grabCut(self.ims,self.mask,None,self.bg,self.fg,iterCount=1,mode=cv2.GC_INIT_WITH_MASK)
        mask2=np.where((self.mask==1)|(self.mask==3),1,0)
        mask2=mask2.astype(np.uint8)
        self.img=self.ims*mask2[:,:,None]


if __name__ == '__main__':
    path=r'./imgs/statue_small.jpg'
    g=Grabcut(path)
    g.run()
        
