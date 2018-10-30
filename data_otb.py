# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-30 10:12:40
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-30 21:20:21
import os
import sys
import math
import torch
import random
import os.path as osp
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
from axis import x1y1x2y2_to_xywh, xywh_to_x1y1x2y2, x1y1wh_to_xywh, x1y1wh_to_x1y1x2y2, point_center_crop, resize, x1y1x2y2x3y3x4y4_to_x1y1wh

#data_dir directly contain img / label
data_dir = './OTB2015/otb15/subset'
interval = 20

list1 = os.listdir(data_dir) #OTB2015下的所有文件夹

#记录每个文件夹中图片的数量,减去interval标明index所能取到的极限
number = []
for item in list1:
    number.append(len(os.listdir(osp.join(data_dir, item, 'img'))))
number = [i-interval for i in number]

#sum1记录index所能取到的总和
#根据sum和num来确定合法index范围
sum1 = [0]
for a in range(len(number)):
    sum1.append(sum1[a]+number[a])

# 序号可以看作是文件夹idx
# number [10, 20, 50, 30]
# sum    [0,  10, 30, 80, 110]
class MyDataset(Dataset):

    def __init__(self, root_dir='./OTB2015/', anchor_scale = 64, k = 5, vis = False):
        self.root_dir = root_dir
        self.anchor_shape = self._get_anchor_shape(anchor_scale)
        self.k = k    
        self.count = 0
        self.vis = vis
        self.save_dir = self._create_vis_tmp_dir()

    def _create_vis_tmp_dir(self):
        """
        创造临时空间查看可视化结果
        """
        workspace = osp.abspath('./')
        path = osp.join(workspace, 'tmp', 'vis_dataloader')
        
        if not os.path.exists(path):
            os.makedirs(path)    

        return path

    def _get_anchor_shape(self, a):
        """
        anchor1: [a * sqrt(3), a / sqrt(3)]
        anchor2: [a * sqrt(2), a / sqrt(2)]
        anchor3: [a, a]
        anchor4: [a / sqrt(2), a * sqrt(2)]
        anchor5: [a / sqrt(3), a * sqrt(3)]
        """
        s = a**2
        r = [
            [3*math.sqrt(s/3.), math.sqrt(s/3.)],  
            [2*math.sqrt(s/2.), math.sqrt(s/2.)], 
            [a,a],                                
            [math.sqrt(s/2.),   2*math.sqrt(s/2.)],
            [math.sqrt(s/3.),3*math.sqrt(s/3.)]   
            ]
        return [list(map(round, i)) for i in r]

    def __len__(self):
        return sum1[-1]
    
    def _which(self, index, sum1):
        """
        从总体中选取index
        看index是来自第几个文件夹
        exp [0, 107, 879, 1084, 1472]
        折半查找
        low -- 0
        high-- 4
        """
        length_sum = len(sum1)

        low = 0

        high = len(sum1) - 1
        while(high - low > 1):
            mid = (high+low) // 2
            if sum1[mid] <= index:
                low = mid
            elif sum1[mid] > index:
                high = mid
        return low
    
    def _trans_tensor_PIL(self, img_tensor):
        """
        3, 127, 127
        """
        assert isinstance(img_tensor, torch.Tensor), 'input should be tensor'
        #img_np = img_tensor.numpy()
        #img_np = img_np.transpose(1,2,0).astype(np.uint8)
        #img_pil= Image.fromarray(img_np)
        img_pil = transforms.ToPILImage()(img_tensor).convert('RGB')
        return img_pil

    def __getitem__(self, index):
        """
        从 index 产生一个模版
        从 index + rand 产生一个detection
        这个detection是包含模版的 可以做可视化

        返回
        template  -- 模版127
        detection -- 包含该模版的图255
        clabel -- 每个anchor的类别 5, 17, 17
        rlable -- 每个anchor的回归 20,17, 17
        """
        low = self._which(index, sum1)
        index -= sum1[low]
        folder = list1[low]
        print('**Train** img folder <==> {:8} index <==> {:3}'.format(folder, index))

        ############# template ##########################
        # 产生对应index的img instance
        img_name_list = sorted(os.listdir(osp.join(self.root_dir, folder, 'img')))
        img = img_name_list[index]
        img = Image.open(osp.join(self.root_dir, folder, 'img', img))
        img = img.convert('RGB') if not img.mode == 'RGB' else img
        
        # 产生对应index的gtbox instance
        gtbox_name_list = sorted(os.listdir(osp.join(self.root_dir, folder, 'label')))
        gtbox = gtbox_name_list[index]  #e.g. 00000090.txt
        with open(osp.join(self.root_dir, folder, 'label', gtbox)) as f:
            gtbox = f.read().strip('\n')
            print(gtbox)
            gtbox = gtbox.split('\t') if gtbox.find(',') == -1 else gtbox.split(',')
        gtbox = [round(float(i)) for i in gtbox]
        gtbox = x1y1wh_to_xywh(gtbox)

        template, _, _ = self._transform(img, gtbox, 1, 127)
        

        ############## detection###### ##################
        #从 1-100 中选取一个奇数 random.randrange(1, 100, 2)
        #从 1-interval中随便选取一个数
        rand = random.randrange(1, interval) 
        img_name_list = sorted(os.listdir(osp.join(self.root_dir, folder, 'img')))
        img = img_name_list[index + rand]
        img = Image.open(osp.join(self.root_dir, folder, 'img', img))
        img = img.convert('RGB') if not img.mode == 'RGB' else img
        
        gtbox_name_list = sorted(os.listdir(osp.join(self.root_dir, folder, 'label')))
        gtbox = gtbox_name_list[index + rand]  #e.g. 00000090.txt
        with open(osp.join(self.root_dir, folder, 'label', gtbox)) as f:
            gtbox = f.read().strip('\n')
            gtbox = gtbox.split('\t') if gtbox.find(',') == -1 else gtbox.split(',')
        gtbox = [round(float(i)) for i in gtbox]
        gtbox = x1y1wh_to_xywh(gtbox)
        detection, pcc, ratio = self._transform(img, gtbox, 2, 255)

        a = (gtbox[2]+gtbox[3]) / 2.
        a = math.sqrt((gtbox[2]+a)*(gtbox[3]+a)) * 2
        gtbox = [127, 127, round(255*gtbox[2]/a), round(255*gtbox[3]/a)] #(center x, y, w, h)

        if self.vis and self.count < 1000:
            path_tmplate = osp.join(self.save_dir, '{:05d}_1_template.jpg'.format(self.count))
            path_detect  = osp.join(self.save_dir, '{:05d}_2_detection.jpg'.format(self.count))
            path_origin  = osp.join(self.save_dir, '{:05d}_3_origin.jpg'.format(self.count))

            template = self._trans_tensor_PIL(template)
            detect = self._trans_tensor_PIL(detection)
            
            # gtbox是在detection 255 * 255 上的中心坐标,w,h
            # origin
            origin = detect.copy()
            draw = ImageDraw.Draw(origin)
            draw.rectangle([gtbox[0]-gtbox[2]//2 , gtbox[1]-gtbox[3]//2, gtbox[0]+gtbox[2]//2, gtbox[1]+gtbox[3]//2])

            template.save(path_tmplate)
            detect.save(path_detect)
            origin.save(path_origin)
            self.count += 1

        # clabel (5, 17, 17)
        # rlabel (20,17, 17)
        clabel, rlabel = self._gtbox_to_label(gtbox)
        return template, detection, clabel, rlabel, torch.from_numpy(np.array(pcc).reshape((1,4))), torch.from_numpy(np.array(ratio).reshape((1,1)))
    

    def _transform(self, img, gtbox, area, size):
        """
        img   -- img是Image图像对象
        gtbox -- 中心点xywh
        area  -- aa box的缩放比例  1
        size  -- 127
        """
        img, pcc = point_center_crop(img, gtbox, area)
        img, ratio = resize(img, size)
        img = F.to_tensor(img)
        #img = F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return img, pcc, ratio
        
    def _gtbox_to_label(self, gtbox):
        """
        只输入一个gtbox
        clabel -- [5, 17, 17]
        rlabel -- [20, 17, 17]
        """
        #把通道信息放在前面
        clabel = np.zeros([5, 17, 17]) - 100
        rlabel = np.zeros([20, 17, 17], dtype = np.float32)

        # 分类gt
        # 返回16pos/48neg index1,index2 pixel, index3 channels
        # [[index1, index2, index3], ..., ] type np ndarray
        # 这里的index3是用来标明scale的
        # pos 为 1， neg 为 0， dont care -100
        pos, neg = self._get_64_anchors(gtbox)
        for i in range(len(pos)):
            clabel[pos[i, 2], pos[i, 0], pos[i, 1]] = 1
        for i in range(len(neg)):
            clabel[neg[i, 2], neg[i, 0], neg[i, 1]] = 0

        # 回归gt
        # pos_coord [num_pos, (x0, y0, w, h)]
        pos_coord = self._anchor_coord(pos)
        channel0 = (gtbox[0] - pos_coord[:, 0]) / pos_coord[:, 2]
        channel1 = (gtbox[1] - pos_coord[:, 1]) / pos_coord[:, 3]
        channel2 = np.array([math.log(i) for i in (gtbox[2] / pos_coord[:, 2]).tolist()])
        channel3 = np.array([math.log(i) for i in (gtbox[3] / pos_coord[:, 3]).tolist()])
        for i in range(len(pos)):
            rlabel[pos[i][2]*4, pos[i][0], pos[i][1]] = channel0[i]
            rlabel[pos[i][2]*4 + 1, pos[i][0], pos[i][1]] = channel1[i]
            rlabel[pos[i][2]*4 + 2, pos[i][0], pos[i][1]] = channel2[i]
            rlabel[pos[i][2]*4 + 3, pos[i][0], pos[i][1]] = channel3[i]
        return torch.Tensor(clabel).long(), torch.Tensor(rlabel).float()
    
    def _anchor_coord(self, pos):
        """
        i in pos: i[0] x, i[1] y, i[2] c -- sacle
        对于pos中的每一个位置 产生5个anchor坐标
        """
        result = np.ndarray([0, 4])
        for i in pos:
            #self.anchor_shape[i[2]][0] 表示anchor scale[c] 的 w
            tmp = [7+15*i[0], 7+15*i[1], self.anchor_shape[i[2]][0], self.anchor_shape[i[2]][1]]
            result = np.concatenate([result, np.array(tmp).reshape([1,4])], axis = 0)
        return result

    def _get_64_anchors(self, gtbox):
        """
        这里可以做优化
        每个像素点5个类别
        这里是对17 * 17 * 5个位置遍历，
        如果这里的iou>0.5则是pos
        如果这里的iou<0.5则是neg
        然后根据iou排序 取pos16 neg48
        sorted()传入reverse = True进行降序排列, 默认是从小到大
        """
        pos = {}
        neg = {}
        for a in range(17):
            for b in range(17):
                for c in range(5):
                    # x0, y0, w, h -> x1y1x2y2
                    anchor = [7+15*a, 7+15*b, self.anchor_shape[c][0], self.anchor_shape[c][1]]
                    anchor = xywh_to_x1y1x2y2(anchor)
                    if anchor[0]>=0 and anchor[1]>=0 and anchor[2]<=255 and anchor[3]<=255:
                        iou = self._IOU(anchor, gtbox)
                        if iou >= 0.5:
                            pos['%d,%d,%d' % (a,b,c)] = iou
                        elif iou <= 0.2:
                            neg['%d,%d,%d' % (a,b,c)] = iou
        # for i in dict
        # i[0] is key
        # i[1] is value 
        # 这里有可能有bug pos有可能不足16
        pos = sorted(pos.items(), key = lambda x:x[1], reverse = True)
        pos = [list(map(int, i[0].split(','))) for i in pos[:16]]
        neg = sorted(neg.items(), key = lambda x:x[1], reverse = True)
        neg = [list(map(int, i[0].split(','))) for i in neg[:(64-len(pos))]]
        return np.array(pos), np.array(neg)

    def _IOU(self, a, b):
        """
        输入的gtbox是(中心坐标，w, h)
        生成的anchor并不是针对原图而是针对255*255
        """
        b = xywh_to_x1y1x2y2(b)
        sa = (a[2] - a[0]) * (a[3] - a[1]) 
        sb = (b[2] - b[0]) * (b[3] - b[1])
        w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        area = w * h 
        return area / (sa + sb - area)


transformed_dataset_train = MyDataset(root_dir = data_dir, vis = True)
train_dataloader = DataLoader(transformed_dataset_train, batch_size=1, shuffle=True, num_workers=0)
dataloader = {'train':train_dataloader, 'valid':train_dataloader}

"""
TODO：
不要把gt放在每个加载图片的中间
可以做成随机包含这个图片的区域

为什么gt不是框住所有的object
"""
if __name__ == '__main__':
    print('Do a test for dataloader')
    length = transformed_dataset_train.__len__()
    for i in range(length):
        transformed_dataset_train.__getitem__(i)



