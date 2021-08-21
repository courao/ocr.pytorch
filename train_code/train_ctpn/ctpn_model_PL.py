# -*- coding:utf-8 -*-
#'''
# Created on 18-12-11 上午10:01
#
# @Author: Greg Gao(laygin)
#'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch import optim


class Initializer:
    def __init__(self):
        pass

    def init_module(m,x):
        print(m,x)
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def __call__(self, net):
        """
        Parameters:
            net: Network
        """
        net.apply(self.init_module)

        return net


class RPN_REGR_Loss(nn.Module):
    def __init__(self, sigma=9.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, input, target):
        """
        smooth L1 loss
        :param input:y_preds
        :param target: y_true
        :return:
        """
        try:
            cls = target[0, :, 0]
            regr = target[0, :, 1:3]
            # apply regression to positive sample
            regr_keep = (cls == 1).nonzero()[:, 0]
            regr_true = regr[regr_keep]
            regr_pred = input[0][regr_keep]
            diff = torch.abs(regr_true - regr_pred)
            less_one = (diff < 1.0 / self.sigma).float()
            loss = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1 - less_one) * (
                diff - 0.5 / self.sigma
            )
            loss = torch.sum(loss, 1)
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)
        except Exception as e:
            print("RPN_REGR_Loss Exception:", e)
            # print(input, target)
            loss = torch.tensor(0.0)

        return loss


class RPN_CLS_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L_cls = nn.CrossEntropyLoss(reduction="none")
        # self.L_regr = nn.SmoothL1Loss()
        # self.L_refi = nn.SmoothL1Loss()
        self.pos_neg_ratio = 3

    def forward(self, input, target):
        if config.OHEM:
            cls_gt = target[0][0]
            num_pos = 0
            loss_pos_sum = 0

            # print(len((cls_gt == 0).nonzero()),len((cls_gt == 1).nonzero()))

            if len((cls_gt == 1).nonzero()) != 0:  # avoid num of pos sample is 0
                cls_pos = (cls_gt == 1).nonzero()[:, 0]
                gt_pos = cls_gt[cls_pos].long()
                cls_pred_pos = input[0][cls_pos]
                # print(cls_pred_pos.shape)
                loss_pos = self.L_cls(cls_pred_pos.view(-1, 2), gt_pos.view(-1))
                loss_pos_sum = loss_pos.sum()
                num_pos = len(loss_pos)

            cls_neg = (cls_gt == 0).nonzero()[:, 0]
            gt_neg = cls_gt[cls_neg].long()
            cls_pred_neg = input[0][cls_neg]

            loss_neg = self.L_cls(cls_pred_neg.view(-1, 2), gt_neg.view(-1))
            loss_neg_topK, _ = torch.topk(
                loss_neg, min(len(loss_neg), config.RPN_TOTAL_NUM - num_pos)
            )
            loss_cls = loss_pos_sum + loss_neg_topK.sum()
            loss_cls = loss_cls / config.RPN_TOTAL_NUM
            return loss_cls
        else:
            y_true = target[0][0]
            cls_keep = (y_true != -1).nonzero()[:, 0]
            cls_true = y_true[cls_keep].long()
            cls_pred = input[0][cls_keep]
            loss = F.nll_loss(
                F.log_softmax(cls_pred, dim=-1), cls_true
            )  # original is sparse_softmax_cross_entropy_with_logits
            # loss = nn.BCEWithLogitsLoss()(cls_pred[:,0], cls_true.float())  # 18-12-8
            loss = (
                torch.clamp(torch.mean(loss), 0, 10)
                if loss.numel() > 0
                else torch.tensor(0.0)
            )
            return loss


class basic_conv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=True,
    ):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class EpochLossCallback(Callback):
    """
logs epoch loss and such
    """
    def on_epoch_start(self, trainer, pl_module):
        pl_module.epoch_loss_cls = 0
        pl_module.epoch_loss_regr = 0
        pl_module.epoch_loss = 0

    def on_epoch_end(self, trainer, pl_module):
        trainer.log(
            {
                "epoch_loss_cls": pl_module.epoch_loss_cls,
                "epoch_loss_regr": pl_module.epoch_loss_regr,
                "epoch_loss": pl_module.epoch_loss,
            }
        )


class CTPN_Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        base_model = models.vgg16(pretrained=False)
        layers = list(base_model.features)[:-1]
        self.base_layers = nn.Sequential(*layers)  # block5_conv3 output
        self.rpn = basic_conv(512, 512, 3, 1, 1, bn=False)
        self.brnn = nn.GRU(512, 128, bidirectional=True, batch_first=True)
        self.lstm_fc = basic_conv(256, 512, 1, 1, relu=True, bn=False)
        self.rpn_class = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)
        self.rpn_regress = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)

        # loss classes
        init = Initializer()
        self.critetion_cls = init(RPN_CLS_Loss())
        self.critetion_regr = init(RPN_REGR_Loss())

        # loss
        self.epoch_loss_cls = 0
        self.epoch_loss_regr = 0
        self.epoch_loss = 0

    def forward(self, x):
        x = self.base_layers(x)
        # rpn
        x = self.rpn(x)  # [b, c, h, w]

        x1 = x.permute(0, 2, 3, 1).contiguous()  # channels last   [b, h, w, c]
        b = x1.size()  # b, h, w, c
        x1 = x1.view(b[0] * b[1], b[2], b[3])

        x2, _ = self.brnn(x1)

        xsz = x.size()
        x3 = x2.view(xsz[0], xsz[2], xsz[3], 256)  # torch.Size([4, 20, 20, 256])

        x3 = x3.permute(0, 3, 1, 2).contiguous()  # channels first [b, c, h, w]
        x3 = self.lstm_fc(x3)
        x = x3

        cls = self.rpn_class(x)
        regr = self.rpn_regress(x)

        cls = cls.permute(0, 2, 3, 1).contiguous()
        regr = regr.permute(0, 2, 3, 1).contiguous()

        cls = cls.view(cls.size(0), cls.size(1) * cls.size(2) * 10, 2)
        regr = regr.view(regr.size(0), regr.size(1) * regr.size(2) * 10, 2)

        return cls, regr

    def training_step(self, batch, batch_idx):
        imgs, clss, regrs = batch
        
        cls, regr = self(imgs)

        # compute losses
        loss_cls = self.critetion_cls(cls, clss)
        loss_regr = self.critetion_regr(regr, regrs)
        loss = loss_cls + loss_regr

        # update epoch loss
        self.epoch_loss_cls += loss_cls
        self.epoch_loss_regr += loss_regr
        self.epoch_loss += loss

        return loss

    def configure_optimizers(self):
        # optimizer
        optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        # scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
