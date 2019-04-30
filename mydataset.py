#!/usr/bin/python
# encoding: utf-8

import random
import math
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms

from PIL import Image,ImageEnhance
import numpy as np
import codecs


def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    random_factor = np.random.randint( 0, 31 ) / 10.  # 随机因子
    color_image = ImageEnhance.Color( image ).enhance( random_factor )  # 调整图像的饱和度
    random_factor = np.random.randint( 10, 21 ) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness( color_image ).enhance( random_factor )  # 调整图像的亮度
    random_factor = np.random.randint( 10, 21 ) / 10.  # 随机因1子
    contrast_image = ImageEnhance.Contrast( brightness_image ).enhance( random_factor )  # 调整图像对比度
    random_factor = np.random.randint( 0, 31 ) / 10.  # 随机因子
    return ImageEnhance.Sharpness( contrast_image ).enhance( random_factor )  # 调整图像锐度


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
        for _i in range( len( im ) ):
            im[_i] += random.gauss( mean, sigma )
        return im

    # 将图像转化成数组
    img = np.asarray( image )
    img.flags.writeable = True  # 将数组改为读写模式
    width, height = img.shape[:2]
    img_r = gaussianNoisy( img[:, :, 0].flatten(), mean, sigma )
    img_g = gaussianNoisy( img[:, :, 1].flatten(), mean, sigma )
    img_b = gaussianNoisy( img[:, :, 2].flatten(), mean, sigma )
    img[:, :, 0] = img_r.reshape( [width, height] )
    img[:, :, 1] = img_g.reshape( [width, height] )
    img[:, :, 2] = img_b.reshape( [width, height] )
    return Image.fromarray( np.uint8( img ) )

def data_tf(img):
    img = randomColor(img)
    # img = randomGaussian(img)
    return img

class MyDataset(Dataset):
    def __init__(self,info_filename,train=True, transform=data_tf,target_transform=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.info_filename = info_filename
        self.train = train
        self.files = list()
        self.labels = list()
        with open(info_filename) as f:
            content = f.readlines()
            for line in content:
                fname = line.split(':')[0]
                if fname[-1]=='g':
                    self.files.append(fname)
                    self.labels.append(line.split(':')[1].split('\n')[0].split('\r')[0])


    def name(self):
        return 'MyDataset'

    def __getitem__(self, index):
        # print(self.files[index])
        # print(self.files[index])
        img = Image.open(self.files[index]).convert('L')
        if self.transform is not None:
            img = self.transform( img )
        # target = torch.zeros(len(self.labels_min))
        # target[self.labels_min.index(self.labels[index])] = 1
        label = self.labels[index]
        if self.target_transform is not None:
            label = self.target_transform( label )
        return (img,label)

    def __len__(self):
        return len(self.labels)


class resizeNormalize2(object):

    def __init__(self, size, interpolation=Image.LANCZOS):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class resizeNormalize(object):
    def    __init__(self, size, interpolation=Image.LANCZOS):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        w,h = self.size
        w0 = img.size[0]
        h0 = img.size[1]
        if w<=(w0/h0*h):
            img = img.resize(self.size, self.interpolation)
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
        else:
            w_real = int(w0/h0*h)
            img = img.resize((w_real,h), self.interpolation)
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
            tmp = torch.zeros([img.shape[0],h,w])
            tmp[:,:,:w_real] = img
            img = tmp

        return img

class resizeNormalize3(object):
    def    __init__(self, size, interpolation=Image.LANCZOS):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        w,h = self.size
        w0 = img.size[0]
        h0 = img.size[1]
        w_resized = int(w0*h/h0)
        img = img.resize((w_resized,h),self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        img_tensor = torch.zeros([math.ceil(w_resized/w),img.shape[0],h,w])
        for i in range(math.ceil(w_resized/w)):
            if (i+1)*w<w_resized:
                img_tensor[i, :, :, :] = img[:, :, i * w:(i + 1) * w]
            else:
                img_tensor[i, :, :, :w_resized-
                                     i*w] = img[:, :, i * w:(i + 1) * w]
        return img_tensor

class resizeNormalize4(object):
    def    __init__(self, size, interpolation=Image.LANCZOS):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        w,h = self.size
        w0 = img.size[0]
        h0 = img.size[1]
        w_resized = int(w0*h/h0)
        img = img.resize((w_resized,h),self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        img_tensor = img.view(1, *img.size())
        return img_tensor

class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


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
