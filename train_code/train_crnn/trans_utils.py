import sys, os
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageChops
import cv2
import numpy as np
# import pyblur
import PIL
from PIL import Image, ImageEnhance
# import 
import abc
import time, datetime, inspect
import hashlib
import json
import math


def rename(filepath):
    # print(f'rename {filepath} to 00X')
    filelist = os.listdir(filepath)
    filelist.sort()
    i = 1
    for filename in filelist:
        if str(filename) == '.DS_Store':
            continue
        ext = filename.split('.')[-1]
        shutil.move(filepath + '/' + filename, filepath + '/' + str(i).zfill(3) + '.' + ext)
        i += 1


def zlog(func):
    def new_fn(*args):
        start = time.time()
        result = func(*args)
        end = time.time()
        duration = end - start
        duration = "%.4f" % duration
        # fulltime = time.strftime("%Y-%m-%d %H:%M:%S %f", time.localtime())
        fulltime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        # print(f'{fulltime} {__file__} {func.__name__}:{inspect.getsourcelines(func)[-1]} cost: {duration}s',
        #       sep=' ', end='\n', file=sys.stdout, flush=False)
        return result

    return new_fn


def getpilimage(image):
    if isinstance(image, PIL.Image.Image):  # or isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
        return image
    elif isinstance(image, np.ndarray):
        return cv2pil(image)


def getcvimage(image):
    if isinstance(image, np.ndarray):
        return image
    elif isinstance(image, PIL.Image.Image):  # or isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
        return pil2cv(image)


def cshowone(image):
    image = getcvimage(image)
    cv2.imshow('tmp', image)
    cv2.waitKey(3000)
    return


def pshowone(image):
    image = getpilimage(image)
    image.show()
    return


def cshowtwo(image1, image2):
    width = 800 / 2
    height = 500 / 2
    image1 = getpilimage(image1)
    image2 = getpilimage(image2)
    h, w = image1.size
    image1 = image1.resize((int(width), int(h * height / w)))
    image2 = image2.resize(image1.size)
    bigimg = Image.new('RGB', (width * 2, image1.size[1]))

    bigimg.paste(image1, (0, 0, image1.size[0], image1.size[1]))
    bigimg.paste(image2, (width, 0, width + image1.size[0], image1.size[1]))
    bigimg = getcvimage(bigimg)
    cshowone(bigimg)
    return


def pshowtwo(image1, image2):
    width = int(800 / 2)
    height = int(500 / 2)
    image1 = getpilimage(image1)
    image2 = getpilimage(image2)
    h, w = image1.size
    image1 = image1.resize((int(width), int(h * height / w)))
    image2 = image2.resize(image1.size)
    bigimg = Image.new('RGB', (width * 2, image1.size[1]))

    bigimg.paste(image1, (0, 0, image1.size[0], image1.size[1]))
    bigimg.paste(image2, (width, 0, width + image1.size[0], image1.size[1]))
    pshowone(bigimg)
    return


def pil2cv(image):
    # assert isinstance(image, PIL.Image.Image) or isinstance(image,
    #                                                         PIL.JpegImagePlugin.JpegImageFile), f'input image type is not PIL.image and is {type(
    #     image)}'
    if len(image.split()) == 1:
        return np.asarray(image)
    elif len(image.split()) == 3:
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    elif len(image.split()) == 4:
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGR)


def cv2pil(image):
    assert isinstance(image, np.ndarray), 'input image type is not cv2'
    if len(image.shape) == 2:
        return Image.fromarray(image)
    elif len(image.shape) == 3:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def rgb2gray(filename):
    im = Image.open(filename).convert('L')
    im.show()

    new_image = Image.new("L", (im.width + 6, im.height + 6), 0)
    out_image = Image.new("L", (im.width + 6, im.height + 6), 0)

    new_image.paste(im, (3, 3, im.width + 3, im.height + 3))

    im = getcvimage(im)
    new_image = getcvimage(new_image)
    out_image = getcvimage(out_image)

    _, thresh = cv2.threshold(new_image, 0, 255, cv2.THRESH_OTSU)
    pshowone(thresh)
    image, contours, hierarchy = cv2.findContours(thresh, 3, 2)
    # cnt = contours[0]
    # hull = cv2.convexHull(cnt)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    print(len(contours))
    cv2.polylines(out_image, contours, True, 255)
    # cv2.fillPoly(image, [cnt], 255)
    image = getpilimage(out_image)
    im = getpilimage(im)
    image = image.crop((3, 3, im.width + 3, im.height + 3))
    # char_color = image.crop((3,3,char_image.width + 3, char_image.height + 3))
    image.show()
    return


def uniqueimg(filepath):
    # print(f'unique {filepath}')
    filepath += '/'
    filelist = os.listdir(filepath)
    filelist.sort()
    i = 1
    for filename in filelist:
        if str(filename) == '.DS_Store':
            continue
        fd = np.array(Image.open(filepath + filename))
        fmd5 = hashlib.md5(fd)
        # print(fmd5.hexdigest())
        # print(filename)
        ext = filename.split('.')[-1]
        shutil.move(filepath + filename, filepath + fmd5.hexdigest() + '.' + ext)
        # i += 1


if __name__ == '__main__':
    # print(sys.argv)
    # rename(sys.argv[1])
    # uniqueimg('/Users/ganyufei/temp/')
    allimg = getlabeljson('/Users/ganyufei/Desktop/jiu_zheng/jiu_zheng.json')
    print(cal_sim_all(allimg['20190113_092023.jpg'], allimg['NID 7333475056 (1) Front.jpg']))
    # print(cal_sim_all(allimg['20190113_092023.jpg'], allimg['20190113_092023.jpg']))

    # genpair(12)









