#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import codecs
import trans

import pytorch_lightning as pl
from torch.utils.data import DataLoader


crop = trans.Crop(probability=0.1)
crop2 = trans.Crop2(probability=1.1)
random_contrast = trans.RandomContrast(probability=0.1)
random_brightness = trans.RandomBrightness(probability=0.1)
random_color = trans.RandomColor(probability=0.1)
random_sharpness = trans.RandomSharpness(probability=0.1)
compress = trans.Compress(probability=0.3)
exposure = trans.Exposure(probability=0.1)
rotate = trans.Rotate(probability=0.1)
blur = trans.Blur(probability=0.1)
salt = trans.Salt(probability=0.1)
adjust_resolution = trans.AdjustResolution(probability=0.1)
stretch = trans.Stretch(probability=0.1)

crop.setparam()
crop2.setparam()
random_contrast.setparam()
random_brightness.setparam()
random_color.setparam()
random_sharpness.setparam()
compress.setparam()
exposure.setparam()
rotate.setparam()
blur.setparam()
salt.setparam()
adjust_resolution.setparam()
stretch.setparam()


def inverse_color(image):
    if np.random.random() < 0.4:
        image = ImageOps.invert(image)
    return image


def data_tf(img):
    """
    transforms data by cropping, contrast, brightness, color, sharpness, and more
    """
    img = crop.process(img)
    img = random_contrast.process(img)
    img = random_brightness.process(img)
    img = random_color.process(img)
    img = random_sharpness.process(img)
    if img.size[1] >= 32:
        img = compress.process(img)
        img = adjust_resolution.process(img)
        img = blur.process(img)
    img = exposure.process(img)
    # img = rotate.process(img)
    img = salt.process(img)
    img = inverse_color(img)
    img = stretch.process(img)
    return img


class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.LANCZOS, is_test=False):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.is_test = is_test

    def __call__(self, img):
        w, h = self.size
        w0 = img.size[0]
        h0 = img.size[1]
        if w <= (w0 / h0 * h):
            img = img.resize(self.size, self.interpolation)
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
        else:
            w_real = int(w0 / h0 * h)
            img = img.resize((w_real, h), self.interpolation)
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
            start = random.randint(0, w - w_real - 1)
            if self.is_test:
                start = 5
                w += 10
            tmp = torch.zeros([img.shape[0], h, w]) + 0.5
            tmp[:, :, start : start + w_real] = img
            img = tmp
        return img


class alignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels


class MyDataset(Dataset):
    """
    This dataset reads data from a text file

    text file contains:


    """
    def __init__(
        self,
        info_filename,
        config,
        train=True,
        transform=data_tf,
        target_transform=None,
        remove_blank=False,
        val_step=False,
    ):
        super().__init__()
        self.config = config
        self.transform = transform
        self.target_transform = target_transform
        self.info_filename = info_filename
        if isinstance(self.info_filename, str):
            self.info_filename = [self.info_filename]
        self.train = train
        self.files = []
        self.labels = []
        for info_name in self.info_filename:

            with open(info_name) as f:
                content = f.readlines()
                for line in content:

                    if r"\t" in line:
                        if len(line.split(r"\t")) != 2:
                            print("abnormal text:", line)
                        fname, label = line.split(r"\t")

                    else:
                        fname, label = line.split("g:")
                        fname += "g"

                    if remove_blank:
                        label = label.strip()
                    else:
                        label = " " + label.strip() + " "

                    self.files.append(fname)
                    self.labels.append(label)

        if val_step == True:
            self.files = self.files[: config.batchSize]
            self.labels = self.labels[: config.batchSize]

    def name(self):
        return "MyDataset"

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        if self.transform is not None:
            img = self.transform(img)
        img = img.convert("L")

        label = self.labels[index]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return (img, label)

    def __len__(self):
        return len(self.labels)


class MyDataModule(pl.LightningDataModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.kwargs = kwargs

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(
            MyDataset(config=self.config, **self.kwargs),
            batch_size=self.config.batchSize,
            shuffle=True,
            num_workers=int(self.config.workers),
            collate_fn=alignCollate(
                imgH=self.config.imgH,
                imgW=self.config.imgW,
                keep_ratio=self.config.keep_ratio,
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            MyDataset(config=self.config, **self.kwargs),
            transform=resizeNormalize((config.imgW, config.imgH), is_test=True),
        )

    def val_dataloader(self):
        return DataLoader(
            MyDataset(config=self.config, val_step=True, **self.kwargs),
            batch_size=self.config.batchSize,
            shuffle=False,
            num_workers=int(self.config.workers),
            collate_fn=alignCollate(
                imgH=self.config.imgH,
                imgW=self.config.imgW,
                keep_ratio=self.config.keep_ratio,
            ),
        )


if __name__ == "__main__":
    pass
