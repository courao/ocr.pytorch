# -*- coding:utf-8 -*-
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
import pytorch_lightning as pl
from torch.utils.data import random_split
import math
from torch.utils.data import DataLoader
from PIL import Image


class ICDARDataset(Dataset):
    def __init__(self, img_names, datadir, labelsdir=None, val_data=False):
        super().__init__()
        self.datadir = datadir
        self.img_names = img_names
        self.labelsdir = labelsdir
        self.val_data = val_data

    def __len__(self):
        return len(self.img_names)

    def box_transfer(self, coor_lists, rescale_fac=1.0):
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
            gtboxes.append((xmin, ymin, xmax, ymax))
        return np.array(gtboxes)

    def box_transfer_v2(self, coor_lists, rescale_fac=1.0):
        """
        This function is used to transfer the coordinates of bboxes from one image to another.
        The input coor_lists is a list of coordinates of bboxes in the source image, which are represented by a list of 4 numbers for each bbox, namely xmin, ymin, xmax, ymax.
        The output gtboxes is a numpy array of coordinates of bboxes in the target image, which are represented by a list of 4 numbers for each bbox, namely xmin, ymin, xmax, ymax.
        The parameter rescale_fac is used to rescale the coordinates of bboxes in the source image before transferring them to the target image.

        Parameters
        ----------
        coor_lists : list
            A list of coordinates of bboxes in the source image.
            Each element in the list is a list of 4 numbers for a bbox, namely xmin, ymin, xmax, ymax.
        rescale_fac : float
            The factor used to rescale the coordinates of bboxes in the source image before transferring them to the target image.

        Returns
        -------
        gtboxes : numpy array
            A numpy array of coordinates of bboxes in the target image.
            Each row in the numpy array represents a bbox, which is represented by a list of 4 numbers for a bbox, namely xmin, ymin, xmax, ymax.

        Examples
        --------
        >>> coor_lists = [[10, 20, 30, 40], [50, 60, 70, 80]]
        >>> gtboxes = box_transfer_v2(coor_lists)
        >>> print(gtboxes)
        [[10 20 30 40]
        [50 60 70 80]]

        >>> coor_lists = [[10, 20, 30, 40], [50, 60, 70, 80]]
        >>> gtboxes = box_transfer_v2(coor_lists, rescale_fac=2)
        >>> print(gtboxes)
        [[ 5 10 15 20]
        [25 30 35 40]]
        """
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
                next = 16 * i - 0.5
                gtboxes.append((prev, ymin, next, ymax))
                prev = next
            gtboxes.append((prev, ymin, xmax, ymax))

        return np.array(gtboxes)

    def parse_gtfile(self, gt_path, rescale_fac=1.0):
        coor_lists = list()
        with open(gt_path) as f:
            content = f.readlines()
            for line in content:
                coor_list = line.split(",")[:8]
                if len(coor_list) == 8:
                    coor_lists.append(coor_list)
        return self.box_transfer_v2(coor_lists, rescale_fac)

    def draw_boxes(self, img, cls, base_anchors, gt_box):
        for i in range(len(cls)):
            if cls[i] == 1:
                pt1 = (int(base_anchors[i][0]), int(base_anchors[i][1]))
                pt2 = (int(base_anchors[i][2]), int(base_anchors[i][3]))
                img = cv2.rectangle(img, pt1, pt2, (200, 100, 100))
        for i in range(gt_box.shape[0]):
            pt1 = (int(gt_box[i][0]), int(gt_box[i][1]))
            pt2 = (int(gt_box[i][2]), int(gt_box[i][3]))
            img = cv2.rectangle(img, pt1, pt2, (100, 200, 100))
        return img

    def __getitem__(self, idx):
        """
        Arguments:
            self: self class object
            idx: index of image in the dataset

        Returns:
            tuple: (image, target) where target is a dictionary containing the following fields
                - boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
                - labels (Int64Tensor[N]): the label for each bounding box. 0 represents always the background class.
                - image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
                - area (Tensor[N]): The area of the bounding box.
                - iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
        """
        img_name = self.img_names[idx]
        img_path = os.path.join(self.datadir, img_name)
        img = cv2.imread(img_path)
        img_copy = Image.open(img_path).convert("RGB")

        #####for read error, use default image#####
        if img is None:
            print(img_path)
            with open("error_imgs.txt", "a") as f:
                f.write("{}\n".format(img_path))
            img_name = "img_2647.jpg"
            img_path = os.path.join(self.datadir, img_name)
            img = cv2.imread(img_path)

        #####for read error, use default image#####
        h, w, c = img.shape
        rescale_fac = max(h, w) / 1600
        if rescale_fac > 1.0:
            h = int(h / rescale_fac)
            w = int(w / rescale_fac)
            img = cv2.resize(img, (w, h))

        if self.val_data:
            img_copy = img_copy.resize((w, h))

        gt_path = os.path.join(self.labelsdir, "gt_" + img_name.split(".")[0] + ".txt")
        gtbox = self.parse_gtfile(gt_path, rescale_fac)

        # 33% chance of clipping image
        if np.random.randint(2) == 1:
            img = img[:, ::-1, :]
            newx1 = w - gtbox[:, 2] - 1
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        [cls, regr], base_anchors = cal_rpn(
            (h, w), (int(h / 16), int(w / 16)), 16, gtbox
        )

        m_img = img - IMAGE_MEAN

        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])

        cls = np.expand_dims(cls, axis=0)

        # transform to torch tensor
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()

        if self.val_data:
            # include image while validating
            # because transforms are irreversible, I think
            return (w, h), img_path, m_img, cls, regr
        else:
            return m_img, cls, regr


class ICDARDataModule(pl.LightningDataModule):
    def __init__(self, config, train_val_split=0.8, **kwargs):
        """
        datadir: image's directory
        labelsdir: annotations' directory
        """
        super().__init__()

        self.config = config
        self.datadir = config.icdar17_mlt_img_dir
        self.labelsdir = config.icdar17_mlt_gt_dir
        self.train_val_split = train_val_split
        self.img_names = [
            file_name
            for file_name in os.listdir(self.datadir)
            if (
                (file_name != ".DS_Store")
                and (os.path.isfile(os.path.join(self.datadir, file_name)))
            )
        ]
        self.kwargs = kwargs

        # check if directory exists
        if not os.path.isdir(self.datadir):
            raise Exception("[ERROR] {} is not a directory".format(datadir))
        if not os.path.isdir(self.labelsdir):
            raise Exception("[ERROR] {} is not a directory".format(labelsdir))

        # split data into train and val
        dataset_size = len(self.img_names)
        train_dataset_size = math.floor(dataset_size * 0.8)
        val_dataset_size = dataset_size - train_dataset_size
        self.train_data, self.val_data = random_split(
            self.img_names, [train_dataset_size, val_dataset_size]
        )

    def collate_fn(self, batch):
        # very important, dataloader will throw errors
        # expects tensors inside each batch to be the same size
        return tuple(zip(*batch))

    def train_dataloader(self):
        dataset = ICDARDataset(
            img_names=self.train_data, datadir=self.datadir, labelsdir=self.labelsdir
        )
        return DataLoader(dataset, **self.kwargs)

    def val_dataloader(self):
        dataset = ICDARDataset(
            img_names=self.val_data,
            datadir=self.datadir,
            labelsdir=self.labelsdir,
            val_data=True,
        )
        # return image data with dataloader since the transforms are irreversible
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=None,
        )

    def test_dataloader(self):
        dataset = ICDARDataset(
            img_names=self.val_data, datadir=self.datadir, labelsdir=self.labelsdir
        )
        return DataLoader(dataset, **self.kwargs)


if __name__ == "__main__":
    xmin = 15
    xmax = 95
    for i in range(xmin // 16 + 1, xmax // 16 + 1):
        print(16 * i - 0.5)
