# -*- coding:utf-8 -*-

#############################
# Created on 18-12-11 10:09 #
#            ###            #
# @Author: Greg Gao(laygin) #
#############################

import os
from getpaths import getpath


base_dir = getpath() / ".." / ".." / "ocr_pytorch_data"

icdar17_mlt_img_dir = base_dir / "ICDAR2017_MLT" / "train/"
icdar17_mlt_gt_dir = base_dir / "ICDAR2017_MLT" / "train_gt/"
num_workers = 0  # because I'm on cpu
pretrained_weights = base_dir / ".." / "checkpoints/CTPN.pth"


anchor_scale = 16
IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7

RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300

# bgr can find from  here: https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py
IMAGE_MEAN = [123.68, 116.779, 103.939]
OHEM = True

checkpoints_dir = base_dir / "checkpoints"
outputs = base_dir / "logs"
