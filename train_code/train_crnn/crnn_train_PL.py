"""
Data comes in in the form of images paired to text files that have:

x1,y1,x2,y2,x3,y3,x4,y4, script_language, transcription


"""


import os
from crnn_data_PL import MyDataModule, resizeNormalize
from crnn_model_PL import CRNN, InitializeWeights, LoadCheckpoint
import config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


def train():

    data = MyDataModule(config=config, info_filename=config.train_infofile)

    model = CRNN(
        config=config,
        imgH=config.imgH,
        nc=config.nc,
        nclass=config.nclass,
        nh=config.nh,
    )

    # load checkpoint if it exists

    trainer = pl.Trainer(
        gpus=0,
        max_epochs=config.max_epochs,
        log_every_n_steps=1,
        callbacks=[
            InitializeWeights(),
            LoadCheckpoint(config.pretrained_model),
            ModelCheckpoint(
                dirpath="/Users/mosaicchurchhtx/Desktop/ocr_pytorch/",
                monitor="batch_loss",
            ),
        ],
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    train()
