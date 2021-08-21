# -*- coding:utf-8 -*-

#############################
# Created on 18-12-27 10:31 #
#            ###            #
# @Author: Greg Gao(laygin) #
#############################

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import argparse

import config
from ctpn_model_PL import CTPN_Model, EpochLossLogger
from dataset_PL import ICDARDataModule


def train():
    random_seed = 2021
    torch.random.manual_seed(random_seed)
    np.random.seed(random_seed)

    epochs = 30
    lr = 1e-3

    datamodule = ICDARDataModule(
        config.icdar17_mlt_img_dir,
        config.icdar17_mlt_gt_dir,
        batch_size=1,
        num_workers=config.num_workers,
        shuffle=True,
    )

    model = CTPN_Model()

    trainer = pl.Trainer(gpus=0)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
