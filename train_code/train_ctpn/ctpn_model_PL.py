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
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch import optim
from torchvision import transforms
import torchvision
from PIL import Image, ImageDraw
import numpy as np
import cv2


try:
    import config as CONFIG
    from ctpn_utils import (
        gen_anchor,
        bbox_transfor_inv,
        clip_box,
        filter_bbox,
        nms,
        TextProposalConnectorOriented,
        resize,
    )
except Exception:
    from train_code.train_ctpn import config as CONFIG
    from train_code.train_ctpn.ctpn_utils import (
        gen_anchor,
        bbox_transfor_inv,
        clip_box,
        filter_bbox,
        nms,
        TextProposalConnectorOriented,
        resize,
    )


def using_gpu():
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
        self.pos_neg_ratio = 3

    def forward(self, input, target):
        if CONFIG.OHEM:
            cls_gt = target[0][0]
            num_pos = 0
            loss_pos_sum = 0

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
                loss_neg, min(len(loss_neg), CONFIG.RPN_TOTAL_NUM - num_pos)
            )
            loss_cls = loss_pos_sum + loss_neg_topK.sum()
            loss_cls = loss_cls / CONFIG.RPN_TOTAL_NUM
            return loss_cls
        else:
            y_true = target[0][0]
            cls_keep = (y_true != -1).nonzero()[:, 0]
            cls_true = y_true[cls_keep].long()
            cls_pred = input[0][cls_keep]
            loss = F.nll_loss(
                F.log_softmax(cls_pred, dim=-1), cls_true
            )  # original is sparse_softmax_cross_entropy_with_logits
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


class LoadCheckpoint(Callback):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path

    def on_pretrain_routine_start(self, trainer, pl_module):
        # load checkpoint if checkpoint is found
        if os.path.isfile(self.checkpoint_path):
            device = torch.device("cuda:0" if using_gpu() else "cpu")
            pl_module.load_state_dict(
                torch.load(self.checkpoint_path, map_location=device)[
                    "model_state_dict"
                ]
            )
            pl_module.to(device)
            pl_module.eval()
        else:
            print("checkpoint not found, skipping checkpoint load step")


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


class LossAndCheckpointCallback(Callback):
    """
    Logs epoch loss and such
    """

    def __init__(self, cfg, len_dataset):
        super().__init__()
        self.cfg = cfg
        self.len_dataset = len_dataset
        self.best_loss_cls = 100
        self.best_loss_regr = 100
        self.best_epoch_loss = 100

    def save_checkpoint(self, state, epoch, loss_cls, loss_regr, loss, ext="pth"):
        check_path = os.path.join(
            CONFIG.checkpoints_dir,
            f"v3_ctpn_ep{epoch:02d}_"
            f"{loss_cls:.4f}_{loss_regr:.4f}_{loss:.4f}.{ext}",
        )

        try:
            torch.save(state, check_path)
        except BaseException as e:
            print(e)
            print("failed to save checkpoint to {}".format(check_path))

        print("saving to {}".format(check_path))

    def on_epoch_start(self, trainer, pl_module):
        pl_module.epoch_loss_cls = 0
        pl_module.epoch_loss_regr = 0
        pl_module.epoch_loss = 0

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch_size = pl_module.config.batch_size
        epoch_loss_cls = pl_module.epoch_loss_cls
        epoch_loss_regr = pl_module.epoch_loss_regr
        epoch_loss = pl_module.epoch_loss
        epoch = trainer.current_epoch

        epoch_loss_cls /= epoch_size
        epoch_loss_regr /= epoch_size
        epoch_loss /= epoch_size

        # log epoch loss
        dict_ = {
            "epoch_loss_cls": epoch_loss_cls,
            "epoch_loss_regr": epoch_loss_regr,
            "epoch_loss": epoch_loss,
        }
        self.log_dict(dict_, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # save checkpoint loss is better than before
        if (
            self.best_loss_cls > epoch_loss_cls
            or self.best_loss_regr > epoch_loss_regr
            or self.best_loss > epoch_loss
        ):
            self.best_loss = epoch_loss
            self.best_loss_regr = epoch_loss_regr
            self.best_loss_cls = epoch_loss_cls

            self.save_checkpoint(
                {"model_state_dict": pl_module.state_dict(), "epoch": epoch},
                epoch,
                self.best_loss_cls,
                self.best_loss_regr,
                self.best_loss,
            )


class CTPN_Model(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        if config == None:
            self.config = CONFIG
        else:
            self.config = config
        base_model = models.vgg16(pretrained=False)
        layers = list(base_model.features)[:-1]
        self.base_layers = nn.Sequential(*layers)  # block5_conv3 output
        self.rpn = basic_conv(512, 512, 3, 1, 1, bn=False)
        self.brnn = nn.GRU(512, 128, bidirectional=True, batch_first=True)
        self.lstm_fc = basic_conv(256, 512, 1, 1, relu=True, bn=False)
        self.rpn_class = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)
        self.rpn_regress = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)

        # loss classes
        self.criterion_cls = RPN_CLS_Loss()  # for classifying which language
        self.criterion_regr = RPN_REGR_Loss()  # for predicting bounding boxes for text

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

    def shared_step(self, batch, batch_idx):
        imgs, clss, regrs = batch

        cls, regr = self(imgs)

        # compute losses
        loss_cls = self.criterion_cls(cls, clss)
        loss_regr = self.criterion_regr(regr, regrs)
        loss = loss_cls + loss_regr

        # log batch loss
        dict_ = {
            "batch_loss_cls": loss_cls,
            "batch_loss_regr": loss_regr,
            "batch_loss": loss,
        }
        self.log_dict(dict_, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # update epoch loss
        self.epoch_loss_cls += loss_cls
        self.epoch_loss_regr += loss_regr
        self.epoch_loss += loss

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def load_checkpoint(self, checkpoint_path=None):
        # grab checkpoint where it normally is
        if checkpoint_path == None:
            checkpoint_path = self.config.pretrained_weights

        if os.path.isfile(checkpoint_path):
            device = torch.device("cuda:0" if using_gpu() else "cpu")
            self.load_state_dict(
                torch.load(checkpoint_path, map_location=device)["model_state_dict"]
            )
            self.to(device)
            self.eval()
        else:
            print("checkpoint not found, skipping checkpoint load step")

    def get_det_boxes(self, image, cls=None, regr=None, display=True, expand=True):
        """
        cls: confidence scores on bounding boxes
        regr: bounding boxes
        """
        if cls == None or regr == None:
            image_copy = image.copy()
            image = image - CONFIG.IMAGE_MEAN
            image = torch.from_numpy(image.transpose([2, 0, 1])).float()
            image = image.unsqueeze(0)
            cls, regr = self(image)
            image = image_copy
        image_r = image.copy()
        image_c = image.copy()
        h, w = image.shape[:2]
        image = image.astype(np.float32) - CONFIG.IMAGE_MEAN
        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()

        cls_prob = F.softmax(cls, dim=-1).cpu().detach().numpy()
        regr = regr.cpu().detach().numpy()
        anchor = gen_anchor((int(h / 16), int(w / 16)), 16)
        bbox = bbox_transfor_inv(anchor, regr)
        bbox = clip_box(bbox, [h, w])

        fg = np.where(cls_prob[0, :, 1] > CONFIG.prob_thresh)[0]
        select_anchor = bbox[fg, :]
        select_score = cls_prob[0, fg, 1]
        select_anchor = select_anchor.astype(np.int32)
        keep_index = filter_bbox(select_anchor, 16)

        # nms: ?
        select_anchor = select_anchor[keep_index]
        select_score = select_score[keep_index]
        select_score = np.reshape(select_score, (select_score.shape[0], 1))
        nmsbox = np.hstack((select_anchor, select_score))
        keep = nms(nmsbox, 0.3)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]

        # text line
        textConn = TextProposalConnectorOriented()
        text = textConn.get_text_lines(select_anchor, select_score, [h, w])

        # expand text
        if expand:
            for idx in range(len(text)):
                text[idx][0] = max(text[idx][0] - 10, 0)
                text[idx][2] = min(text[idx][2] + 10, w - 1)
                text[idx][4] = max(text[idx][4] - 10, 0)
                text[idx][6] = min(text[idx][6] + 10, w - 1)

        if display:
            blank = np.zeros(image_c.shape, dtype=np.uint8)
            for box in select_anchor:
                pt1 = (box[0], box[1])
                pt2 = (box[2], box[3])
                blank = cv2.rectangle(blank, pt1, pt2, (50, 0, 0), -1)
            image_c = image_c + blank
            image_c[image_c > 255] = 255
            for i in text:
                s = str(round(i[-1] * 100, 2)) + "%"
                i = [int(j) for j in i]
                cv2.line(image_c, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)
                cv2.line(image_c, (i[0], i[1]), (i[4], i[5]), (0, 0, 255), 2)
                cv2.line(image_c, (i[6], i[7]), (i[2], i[3]), (0, 0, 255), 2)
                cv2.line(image_c, (i[4], i[5]), (i[6], i[7]), (0, 0, 255), 2)
                cv2.putText(
                    image_c,
                    s,
                    (i[0] + 13, i[1] + 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        #      text_rectangles, img_framed, image
        return text, image_c, image_r

    def validation_step(self, batch, batch_idx):
        # visualize bounding boxes and text

        resize_dim, img_path, images, real_cls, real_regr = batch

        pred_cls, pred_regr = self(images)

        # visualize data
        grid = []

        to_tensor = transforms.ToTensor()

        for i in range(self.config.batch_size):
            w, h = resize_dim
            w = w.item()
            h = h.item()
            img_copy = np.array(Image.open(img_path[0]).resize((w, h)))
            text_recs, img_framed, images = self.get_det_boxes(
                img_copy, pred_cls, pred_regr
            )

            tensor = torch.stack([to_tensor(img_framed)])
            # tensor = (tensor + 1) / 2
            grid.append(tensor)

        # combine images to one image
        grid = torchvision.utils.make_grid(torch.cat(grid, 0), 1)

        self.logger.experiment.add_image(
            "image --> predicted text", grid, self.current_epoch, dataformats="CHW"
        )

        return

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def configure_optimizers(self):
        # optimizer
        optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        # scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
