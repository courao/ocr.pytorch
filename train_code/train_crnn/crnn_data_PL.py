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


debug_idx = 0
debug = True

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


def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    random_factor = np.random.randint(0, 31) / 10.0  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.0  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(
        random_factor
    )  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.0  # 随机因1子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(
        random_factor
    )  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.0  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度


def randomGaussian(image, mean=0.2, sigma=0.3):
    """
     对图像进行高斯噪声处理
    :param image:
    :return:
    """

    def gaussianNoisy(im, mean=0.2, sigma=0.3):
        """
        对图像做高斯噪音处理
        :param im: 单通道图像
        :param mean: 偏移量
        :param sigma: 标准差
        :return:
        """
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    # 将图像转化成数组
    img = np.asarray(image)
    img.flags.writeable = True  # 将数组改为读写模式
    width, height = img.shape[:2]
    img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def inverse_color(image):
    if np.random.random() < 0.4:
        image = ImageOps.invert(image)
    return image


def data_tf(img):
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
    if debug and np.random.random() < 0.001:
        global debug_idx
        img.save("debug_files/{:05}.jpg".format(debug_idx))
        debug_idx += 1
        if debug_idx == 10000:
            debug_idx = 0
    return img


def data_tf_fullimg(img, loc):
    left, top, right, bottom = loc
    img = crop2.process([img, left, top, right, bottom])
    img = random_contrast.process(img)
    img = random_brightness.process(img)
    img = random_color.process(img)
    img = random_sharpness.process(img)
    img = compress.process(img)
    img = exposure.process(img)
    # img = rotate.process(img)
    img = blur.process(img)
    img = salt.process(img)
    # img = inverse_color(img)
    img = adjust_resolution.process(img)
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


class MyDataset(Dataset):
    def __init__(
        self,
        info_filename,
        train=True,
        transform=data_tf,
        target_transform=None,
        remove_blank=False,
    ):
        super(Dataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.info_filename = info_filename
        if isinstance(self.info_filename, str):
            self.info_filename = [self.info_filename]
        self.train = train
        self.files = list()
        self.labels = list()
        for info_name in self.info_filename:
            with open(info_name) as f:
                content = f.readlines()
                for line in content:
                    if "\t" in line:
                        if len(line.split("\t")) != 2:
                            print(line)
                        fname, label = line.split("\t")

                    else:
                        fname, label = line.split("g:")
                        fname += "g"
                    if remove_blank:
                        label = label.strip()
                    else:
                        label = " " + label.strip() + " "
                    self.files.append(fname)
                    self.labels.append(label)

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

    def setup(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            MyDataset(**self.kwargs),
            batch_size=self.config.batchSize,
            shuffle=True,
            num_workers=int(self.config.workers),
            collate_fn=crnn_data_PL.alignCollate(
                imgH=self.config.imgH,
                imgW=self.config.imgW,
                keep_ratio=self.config.keep_ratio,
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            MyDataset(**self.kwargs),
            transform=resizeNormalize((config.imgW, config.imgH), is_test=True),
        )

    def val_dataloader(self):
        return DataLoader(MyDataset(**self.kwargs))


if __name__ == "__main__":
    pass
