import os
from crnn_data_PL import MyDataModule, resizeNormalize
from crnn_model_PL import CRNN, InitializeWeights
import config
import pytorch_lightning as pl


def train():

    data = MyDataModule(config=config, info_filename=config.train_infofile)

    model = CRNN(config, config.imgH, config.nc, config.nclass, config.nh)

    # load checkpoint if it exists
    if os.path.exists(config.pretrained_model):
        model.load_from_checkpoint(config.pretrained_model)

    trainer = pl.Trainer(
        gpus=0,
        max_epochs=config.max_epochs,
        log_every_n_steps=1,
        callbacks=[InitializeWeights()],
    )

    trainer.fit(data, model)


if __name__ == "__main__":
    train()
