#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
from PIL import Image,ImageEnhance,ImageOps
import numpy as np
import codecs
import trans

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

def inverse_color(image):
    if np.random.random()<0.4:
        image = ImageOps.invert(image)
    return image

# def data_tf(img):
#     img = randomColor(img)
#     # img = randomGaussian(img)
#     img = inverse_color(img)
#     return img

def data_tf(img):
    img = crop.process(img)
    img = random_contrast.process(img)
    img = random_brightness.process(img)
    img = random_color.process(img)
    img = random_sharpness.process(img)
    if img.size[1]>=32:
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
        img.save('debug_files/{:05}.jpg'.format(debug_idx))
        debug_idx += 1
        if debug_idx == 10000:
            debug_idx = 0
    return img

def data_tf_fullimg(img,loc):
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



class MyDataset(Dataset):
    def __init__(self,info_filename,train=True, transform=data_tf,target_transform=None,remove_blank = False):
        super(Dataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.info_filename = info_filename
        if isinstance(self.info_filename,str):
            self.info_filename = [self.info_filename]
        self.train = train
        self.files = list()
        self.labels = list()
        for info_name in self.info_filename:
            with open(info_name) as f:
                content = f.readlines()
                for line in content:
                    if '\t' in line:
                        if len(line.split('\t'))!=2:
                            print(line)
                        fname, label = line.split('\t')

                    else:
                        fname,label = line.split('g:')
                        fname += 'g'
                    if remove_blank:
                        label = label.strip()
                    else:
                        label = ' '+label.strip()+' '
                    self.files.append(fname)
                    self.labels.append(label)

    def name(self):
        return 'MyDataset'

    def __getitem__(self, index):
        # print(self.files[index])
        # print(self.files[index])
        img = Image.open(self.files[index])
        if self.transform is not None:
            img = self.transform( img )
        img = img.convert('L')
        # target = torch.zeros(len(self.labels_min))
        # target[self.labels_min.index(self.labels[index])] = 1
        label = self.labels[index]
        if self.target_transform is not None:
            label = self.target_transform( label )
        return (img,label)

    def __len__(self):
        return len(self.labels)

class MyDatasetPro(Dataset):
    def __init__(self, info_filename_txtline=list(), info_filename_fullimg=list(), train=True, txtline_transform=data_tf,
                 fullimg_transform = data_tf_fullimg, target_transform=None):
        super(Dataset, self).__init__()
        self.txtline_transform = txtline_transform
        self.fullimg_transform = fullimg_transform
        self.target_transform = target_transform
        self.info_filename_txtline = info_filename_txtline
        self.info_filename_fullimg = info_filename_fullimg
        if isinstance(self.info_filename_txtline,str):
            self.info_filename_txtline = [self.info_filename_txtline]
        if isinstance(self.info_filename_fullimg,str):
            self.info_filename_fullimg = [self.info_filename_fullimg]
        self.train = train
        self.files = list()
        self.labels = list()
        self.locs = list()
        for info_name in self.info_filename_txtline:
            with open(info_name) as f:
                content = f.readlines()
                for line in content:
                    fname,label = line.split('g:')
                    fname += 'g'
                    label = label.replace('\r','').replace('\n','')
                    self.files.append(fname)
                    self.labels.append(label)
        self.txtline_len = len(self.labels)
        for info_name in self.info_filename_fullimg:
            with open(info_name) as f:
                content = f.readlines()
                for line in content:
                    fname,label,left, top, right, bottom = line.strip().split('\t')
                    self.files.append(fname)
                    self.labels.append(label)
                    self.locs.append([int(left),int(top),int(right),int(bottom)])
        print(len(self.labels),len(self.files))
    def name(self):
        return 'MyDatasetPro'

    def __getitem__(self, index):
        # print(self.files[index])
        label = self.labels[index]
        if self.target_transform is not None:
            label = self.target_transform(label)
        img = Image.open(self.files[index])
        if index>=self.txtline_len:
            # print('fullimg:{}'.format(self.files[index]),img.size)
            img = self.fullimg_transform(img,self.locs[index-self.txtline_len])
            if index%100 == 0:
                img.save('test_imgs/debug-{}-{}.jpg'.format(index,label.strip()))  #debug
        else:
            if self.txtline_transform is not None:
                img = self.txtline_transform(img)
        img = img.convert('L')
        # target = torch.zeros(len(self.labels_min))
        # target[self.labels_min.index(self.labels[index])] = 1
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
    def __init__(self, size, interpolation=Image.LANCZOS,is_test=False):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.is_test = is_test

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
            start = random.randint(0,w-w_real-1)
            if self.is_test:
                start = 5
                w+=10
            tmp = torch.zeros([img.shape[0], h, w])+0.5
            tmp[:,:,start:start+w_real] = img
            img = tmp
        return img

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



if __name__ == '__main__':
    import os
    path = 'images'
    files = os.listdir(path)
    idx = 0
    for f in files:
        img_name = os.path.join(path,f)
        img = Image.open(img_name)
        img.show()
        idx+=1
        if idx>5:
            break