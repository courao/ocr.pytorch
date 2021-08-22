import torch.nn as nn
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.nn import CTCLoss
from torch.autograd import Variable
from utils import strLabelConverter
from torch import optim
import torch
import os


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super().__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output


class InitializeWeights(Callback):
    """
    Initialize weights before training starts
    """

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def on_train_start(self, trainer, pl_module):
        pl_module.apply(self.weights_init)


class LoadCheckpoint(Callback):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path

    def using_gpu(self):
        # torch.cuda.is_available() fails on MAC
        try:
            torch.cuda.current_device()
            torch.cuda.is_available()
            # GPU is used!
            return True
        except AssertionError:
            # GPU is not being used!
            return False
        except AttributeError:
            # GPU is not being used!
            return False
        except RuntimeError:
            # GPU is not being used!
            return False

    def on_pretrain_routine_start(self, trainer, pl_module):
        # load checkpoint if checkpoint is found
        if os.path.isfile(self.checkpoint_path):
            device = torch.device("cuda:0" if self.using_gpu() else "cpu")
            pl_module.load_state_dict(
                torch.load(self.checkpoint_path, map_location="cpu")
            )
            pl_module.to(device)
            pl_module.eval()
        else:
            print("checkpoint not found, skipping checkpoint load step")


class CRNN(pl.LightningModule):
    def __init__(self, config, imgH, nc, nclass, nh, leakyRelu=False):
        super().__init__()
        self.config = config

        assert imgH % 16 == 0, "imgH has to be a multiple of 16"

        # 1x32x128
        self.conv1 = nn.Conv2d(nc, 64, 3, 1, 1)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 64x16x64
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 128x8x32
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))

        # 256x4x16
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))

        # 512x2x16
        self.conv5 = nn.Conv2d(512, 512, 2, 1, 0)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(True)

        # 512x1x16
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh), BidirectionalLSTM(nh, nh, nclass)
        )

        # encoder
        self.converter = strLabelConverter(config.alphabet)

        # loss
        self.criterion = CTCLoss(reduction="sum", zero_infinity=True)

    def forward(self, input):
        # conv features
        x = self.pool1(self.relu1(self.conv1(input)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(
            self.relu3_2(self.conv3_2(self.relu3_1(self.bn3(self.conv3_1(x)))))
        )
        x = self.pool4(
            self.relu4_2(self.conv4_2(self.relu4_1(self.bn4(self.conv4_1(x)))))
        )
        conv = self.relu5(self.bn5(self.conv5(x)))

        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output

    def shared_step(self, batch, batch_idx):
        data = batch
        image, texts = data
        batch_size = image.size(0)

        text, length = self.converter.encode(texts)

        preds = self(image)  # seqLength x batchSize x alphabet_size
        preds_size = Variable(
            torch.IntTensor([preds.size(0)] * batch_size)
        )  # seqLength x batchSize

        # compute and log loss
        loss = (
            self.criterion(preds.log_softmax(2), text, preds_size, length) / batch_size
        )
        self.log(
            "batch_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def configure_optimizers(self):
        if self.config.adam:
            optimizer = optim.Adam(
                self.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999)
            )
        elif self.config.adadelta:
            optimizer = optim.Adadelta(self.parameters(), lr=self.config.lr)
        else:
            optimizer = optim.RMSprop(self.parameters(), lr=self.config.lr)

        return optimizer


if __name__ == "__main__":
    pass
