import torch.nn as nn

# import torchvision.models as models
import torch, os
from PIL import Image
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import random
from recognize.crnn import CRNN
from recognize import config

# copy from mydataset
class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.LANCZOS, is_test=True):
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
            tmp = torch.zeros([img.shape[0], h, w])
            start = random.randint(0, w - w_real - 1)
            if self.is_test:
                start = 0
            tmp[:, :, start : start + w_real] = img
            img = tmp
        return img


# copy from utils
class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + "_"  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    # print(self.dict)
    def encode(self, text):
        length = []
        result = []
        for item in text:
            item = item.decode("utf-8", "strict")
            length.append(len(item))
            for char in item:
                if char not in self.dict.keys():
                    index = 0
                else:
                    index = self.dict[char]
                result.append(index)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert (
                t.numel() == length
            ), "text with length: {} does not match declared length: {}".format(
                t.numel(), length
            )
            if raw:
                return "".join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return "".join(char_list)
        else:
            # batch mode
            assert (
                t.numel() == length.sum()
            ), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum()
            )
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(t[index : index + l], torch.IntTensor([l]), raw=raw)
                )
                index += l
            return texts


# recognize api
class PytorchOcr:
    def __init__(self, model_path="checkpoints/CRNN-1010.pth"):
        alphabet_unicode = config.alphabet_v2
        self.alphabet = "".join([chr(uni) for uni in alphabet_unicode])
        # print(len(self.alphabet))
        self.nclass = len(self.alphabet) + 1
        self.model = CRNN(config.imgH, 1, self.nclass, 256)
        self.cuda = False
        if torch.cuda.is_available():
            self.cuda = True
            self.model.cuda()
            self.model.load_state_dict(
                {k.replace("module.", ""): v for k, v in torch.load(model_path).items()}
            )
        else:
            # self.model = nn.DataParallel(self.model)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.converter = strLabelConverter(self.alphabet)

    def recognize(self, img):
        h, w = img.shape[:2]
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = Image.fromarray(img)
        transformer = resizeNormalize((int(w / h * 32), 32))
        image = transformer(image)
        image = image.view(1, *image.size())
        image = Variable(image)

        if self.cuda:
            image = image.cuda()

        preds = self.model(image)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        txt = self.converter.decode(preds.data, preds_size.data, raw=False).strip()

        return txt


if __name__ == "__main__":
    model_path = "./recognize/crnn_models/CRNN-1008.pth"
    recognizer = PytorchOcr(model_path)
    img_name = "t1.jpg"
    img = cv2.imread(img_name)
    h, w = img.shape[:2]
    res = recognizer.recognize(img)
    print(res)
