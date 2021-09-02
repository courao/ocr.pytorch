# -*- coding:utf-8 -*-
#'''
# Created on 18-12-11 上午10:09
#
# @Author: Greg Gao(laygin)
#'''
import os
from pathed import filedir

# base_dir = 'path to dataset base dir'
base_dir = filedir / ".." / ".."


icdar17_mlt_img_dir = base_dir / "train"
icdar17_mlt_gt_dir = base_dir / "train_gt"
num_workers = 0  # change to 2 when on GPU
pretrained_weights = base_dir / "checkpoints" / "CTPN.pth"
batch_size = 1

max_epochs = 2  # change to 30 when on GPU
anchor_scale = 16
IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7

RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300

# bgr can find from  here: https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py
IMAGE_MEAN = [123.68, 116.779, 103.939]
OHEM = True

prob_thresh = 0.5
height = 720

checkpoints_dir = "./checkpoints"
