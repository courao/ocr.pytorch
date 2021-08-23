# -*- coding:utf-8 -*-

#############################
# Created on 18-12-27 10:31 #
#            ###            #
# @Author: Greg Gao(laygin) #
#############################

import os
import torch
import numpy as np

import config
from ctpn_model_PL import CTPN_Model
from ctpn_model_PL import LossAndCheckpointCallback, InitializeWeights, LoadCheckpoint
from ctpn_data_PL import ICDARDataModule
import pytorch_lightning as pl


def train():
    random_seed = 2021
    torch.random.manual_seed(random_seed)
    np.random.seed(random_seed)

    datamodule = ICDARDataModule(
        config=config,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )

    len_train_dataset = len(datamodule.train_data)

    model = CTPN_Model(config=config)

    trainer = pl.Trainer(
        gpus=0,  # number of gpus, 0 if you want to use cpu
        max_epochs=config.max_epochs,
        log_every_n_steps=1,
        callbacks=[
            LoadCheckpoint(config.pretrained_weights),
            InitializeWeights(),
            LossAndCheckpointCallback(config, len_train_dataset),
        ],
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
