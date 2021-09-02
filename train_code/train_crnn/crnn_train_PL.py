"""
Data comes in in the form of images paired to text files that have:

x1,y1,x2,y2,x3,y3,x4,y4, script_language, transcription


"""


import os
import config
from pathed import filedir
import pytorch_lightning as pl
from crnn_data_PL import MyDataModule, resizeNormalize
from pytorch_lightning.callbacks import ModelCheckpoint
from crnn_model_PL import CRNN, InitializeWeights, LoadCheckpoint


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
    desktop = filedir / ".." / ".." / ".."
    trainer = pl.Trainer(
        gpus=0,
        max_epochs=config.max_epochs,
        log_every_n_steps=1,
        callbacks=[
            InitializeWeights(),
            LoadCheckpoint(config.pretrained_model),
            ModelCheckpoint(
                dirpath=str(desktop / "ocr_pytorch"),
                monitor="batch_loss",
            ),
        ],
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    train()
