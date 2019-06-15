#-*- coding:utf-8 -*-
#'''
# Created on 18-12-27 上午10:34
#
# @Author: Greg Gao(laygin)
#'''

import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
from config import IMAGE_MEAN
from ctpn_utils import cal_rpn


def readxml(path):
    gtboxes = []
    imgfile = ''
    xml = ET.parse(path)
    for elem in xml.iter():
        if 'filename' in elem.tag:
            imgfile = elem.text
        if 'object' in elem.tag:
            for attr in list(elem):
                if 'bndbox' in attr.tag:
                    xmin = int(round(float(attr.find('xmin').text)))
                    ymin = int(round(float(attr.find('ymin').text)))
                    xmax = int(round(float(attr.find('xmax').text)))
                    ymax = int(round(float(attr.find('ymax').text)))

                    gtboxes.append((xmin, ymin, xmax, ymax))

    return np.array(gtboxes), imgfile


# for ctpn text detection
class VOCDataset(Dataset):
    def __init__(self,
                 datadir,
                 labelsdir):
        '''

        :param txtfile: image name list text file
        :param datadir: image's directory
        :param labelsdir: annotations' directory
        '''
        if not os.path.isdir(datadir):
            raise Exception('[ERROR] {} is not a directory'.format(datadir))
        if not os.path.isdir(labelsdir):
            raise Exception('[ERROR] {} is not a directory'.format(labelsdir))

        self.datadir = datadir
        self.img_names = os.listdir(self.datadir)
        self.labelsdir = labelsdir

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.datadir, img_name)
        print(img_path)
        xml_path = os.path.join(self.labelsdir, img_name.replace('.jpg', '.xml'))
        gtbox, _ = readxml(xml_path)
        img = cv2.imread(img_path)
        h, w, c = img.shape

        # clip image
        if np.random.randint(2) == 1:
            img = img[:, ::-1, :]
            newx1 = w - gtbox[:, 2] - 1
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        [cls, regr], _ = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)

        m_img = img - IMAGE_MEAN

        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])

        cls = np.expand_dims(cls, axis=0)

        # transform to torch tensor
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()

        return m_img, cls, regr

class ICDARDataset(Dataset):
    def __init__(self,
                 datadir,
                 labelsdir):
        '''

        :param txtfile: image name list text file
        :param datadir: image's directory
        :param labelsdir: annotations' directory
        '''
        if not os.path.isdir(datadir):
            raise Exception('[ERROR] {} is not a directory'.format(datadir))
        if not os.path.isdir(labelsdir):
            raise Exception('[ERROR] {} is not a directory'.format(labelsdir))

        self.datadir = datadir
        self.img_names = os.listdir(self.datadir)
        self.labelsdir = labelsdir

    def __len__(self):
        return len(self.img_names)

    def box_transfer(self,coor_lists,rescale_fac = 1.0):
        gtboxes = []
        for coor_list in coor_lists:
            coors_x = [int(coor_list[2*i]) for i in range(4)]
            coors_y = [int(coor_list[2*i+1]) for i in range(4)]
            xmin = min(coors_x)
            xmax = max(coors_x)
            ymin = min(coors_y)
            ymax = max(coors_y)
            if rescale_fac>1.0:
                xmin = int(xmin / rescale_fac)
                xmax = int(xmax / rescale_fac)
                ymin = int(ymin / rescale_fac)
                ymax = int(ymax / rescale_fac)
            gtboxes.append((xmin, ymin, xmax, ymax))
        return np.array(gtboxes)

    def box_transfer_v2(self,coor_lists,rescale_fac = 1.0):
        gtboxes = []
        for coor_list in coor_lists:
            coors_x = [int(coor_list[2 * i]) for i in range(4)]
            coors_y = [int(coor_list[2 * i + 1]) for i in range(4)]
            xmin = min(coors_x)
            xmax = max(coors_x)
            ymin = min(coors_y)
            ymax = max(coors_y)
            if rescale_fac > 1.0:
                xmin = int(xmin / rescale_fac)
                xmax = int(xmax / rescale_fac)
                ymin = int(ymin / rescale_fac)
                ymax = int(ymax / rescale_fac)
            prev = xmin
            for i in range(xmin // 16 + 1, xmax // 16 + 1):
                next = 16*i-0.5
                gtboxes.append((prev, ymin, next, ymax))
                prev = next
            gtboxes.append((prev, ymin, xmax, ymax))
        return np.array(gtboxes)

    def parse_gtfile(self,gt_path,rescale_fac = 1.0):
        coor_lists = list()
        with open(gt_path) as f:
            content = f.readlines()
            for line in content:
                coor_list = line.split(',')[:8]
                if len(coor_list)==8:
                    coor_lists.append(coor_list)
        return self.box_transfer_v2(coor_lists,rescale_fac)

    def draw_boxes(self,img,cls,base_anchors,gt_box):
        for i in range(len(cls)):
            if cls[i]==1:
                pt1 = (int(base_anchors[i][0]),int(base_anchors[i][1]))
                pt2 = (int(base_anchors[i][2]),int(base_anchors[i][3]))
                img = cv2.rectangle(img,pt1,pt2,(200,100,100))
        for i in range(gt_box.shape[0]):
            pt1 = (int(gt_box[i][0]),int(gt_box[i][1]))
            pt2 = (int(gt_box[i][2]),int(gt_box[i][3]))
            img = cv2.rectangle(img, pt1, pt2, (100, 200, 100))
        return img

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.datadir, img_name)
        # print(img_path)
        img = cv2.imread(img_path)
        #####for read error, use default image#####
        if img is None:
            print(img_path)
            with open('error_imgs.txt','a') as f:
                f.write('{}\n'.format(img_path))
            img_name = 'img_2647.jpg'
            img_path = os.path.join(self.datadir, img_name)
            img = cv2.imread(img_path)

        #####for read error, use default image#####

        h, w, c = img.shape
        rescale_fac = max(h, w) / 1600
        if rescale_fac>1.0:
            h = int(h/rescale_fac)
            w = int(w/rescale_fac)
            img = cv2.resize(img,(w,h))

        gt_path = os.path.join(self.labelsdir, 'gt_'+img_name.split('.')[0]+'.txt')
        gtbox = self.parse_gtfile(gt_path,rescale_fac)

        # clip image
        if np.random.randint(2) == 1:
            img = img[:, ::-1, :]
            newx1 = w - gtbox[:, 2] - 1
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        [cls, regr], base_anchors = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)
        # debug_img = self.draw_boxes(img.copy(),cls,base_anchors,gtbox)
        # cv2.imwrite('debug/{}'.format(img_name),debug_img)
        m_img = img - IMAGE_MEAN

        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])

        cls = np.expand_dims(cls, axis=0)

        # transform to torch tensor
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()

        return m_img, cls, regr

if __name__ == '__main__':
    xmin = 15
    xmax = 95
    for i in range(xmin//16+1,xmax//16+1):
        print(16*i-0.5)