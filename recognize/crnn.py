import torch.nn as nn
from collections import OrderedDict

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

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
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))


    def forward(self, input):
        # conv features
        x = self.pool1(self.relu1(self.conv1(input)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3_2(self.conv3_2(self.relu3_1(self.bn3(self.conv3_1(x))))))
        x = self.pool4(self.relu4_2(self.conv4_2(self.relu4_1(self.bn4(self.conv4_1(x))))))
        conv = self.relu5(self.bn5(self.conv5(x)))
        # print(conv.size())

        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output


class CRNN_v2(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, leakyRelu=False):
        super(CRNN_v2, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        # 1x32x128
        self.conv1_1 = nn.Conv2d(nc, 32, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.relu1_1 = nn.ReLU(True)

        self.conv1_2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 64x16x64
        self.conv2_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.relu2_1 = nn.ReLU(True)

        self.conv2_2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 128x8x32
        self.conv3_1 = nn.Conv2d(128, 96, 3, 1, 1)
        self.bn3_1 = nn.BatchNorm2d(96)
        self.relu3_1 = nn.ReLU(True)

        self.conv3_2 = nn.Conv2d(96, 192, 3, 1, 1)
        self.bn3_2 = nn.BatchNorm2d(192)
        self.relu3_2 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))

        # 192x4x32
        self.conv4_1 = nn.Conv2d(192, 128, 3, 1, 1)
        self.bn4_1 = nn.BatchNorm2d(128)
        self.relu4_1 = nn.ReLU(True)
        self.conv4_2 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.relu4_2 = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))

        # 256x2x32
        self.bn5 = nn.BatchNorm2d(256)


        # 256x2x32

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))


    def forward(self, input):
        # conv features
        x = self.pool1(self.relu1_2(self.bn1_2(self.conv1_2(self.relu1_1(self.bn1_1(self.conv1_1(input)))))))
        x = self.pool2(self.relu2_2(self.bn2_2(self.conv2_2(self.relu2_1(self.bn2_1(self.conv2_1(x)))))))
        x = self.pool3(self.relu3_2(self.bn3_2(self.conv3_2(self.relu3_1(self.bn3_1(self.conv3_1(x)))))))
        x = self.pool4(self.relu4_2(self.bn4_2(self.conv4_2(self.relu4_1(self.bn4_1(self.conv4_1(x)))))))
        conv = self.bn5(x)
        # print(conv.size())

        b, c, h, w = conv.size()
        assert h == 2, "the height of conv must be 2"
        conv = conv.reshape([b,c*h,w])
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output


def conv3x3(nIn, nOut, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d( nIn, nOut, kernel_size=3, stride=stride, padding=1, bias=False )


class basic_res_block(nn.Module):

    def __init__(self, nIn, nOut, stride=1, downsample=None):
        super( basic_res_block, self ).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3( nIn, nOut, stride )
        m['bn1'] = nn.BatchNorm2d( nOut )
        m['relu1'] = nn.ReLU( inplace=True )
        m['conv2'] = conv3x3( nOut, nOut )
        m['bn2'] = nn.BatchNorm2d( nOut )
        self.group1 = nn.Sequential( m )

        self.relu = nn.Sequential( nn.ReLU( inplace=True ) )
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample( x )
        else:
            residual = x
        out = self.group1( x ) + residual
        out = self.relu( out )
        return out


class CRNN_res(nn.Module):

    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN_res, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.conv1 = nn.Conv2d(nc, 64, 3, 1, 1)
        self.relu1 = nn.ReLU(True)
        self.res1 = basic_res_block(64, 64)
        # 1x32x128

        down1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),nn.BatchNorm2d(128))
        self.res2_1 = basic_res_block( 64, 128, 2, down1 )
        self.res2_2 = basic_res_block(128,128)
        # 64x16x64

        down2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),nn.BatchNorm2d(256))
        self.res3_1 = basic_res_block(128, 256, 2, down2)
        self.res3_2 = basic_res_block(256, 256)
        self.res3_3 = basic_res_block(256, 256)
        # 128x8x32

        down3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=(2, 1), bias=False),nn.BatchNorm2d(512))
        self.res4_1 = basic_res_block(256, 512, (2, 1), down3)
        self.res4_2 = basic_res_block(512, 512)
        self.res4_3 = basic_res_block(512, 512)
        # 256x4x16

        self.pool = nn.AvgPool2d((2, 2), (2, 1), (0, 1))
        # 512x2x16

        self.conv5 = nn.Conv2d(512, 512, 2, 1, 0)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(True)
        # 512x1x16

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        x = self.res1(self.relu1(self.conv1(input)))
        x = self.res2_2(self.res2_1(x))
        x = self.res3_3(self.res3_2(self.res3_1(x)))
        x = self.res4_3(self.res4_2(self.res4_1(x)))
        x = self.pool(x)
        conv = self.relu5(self.bn5(self.conv5(x)))
        # print(conv.size())
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output

if __name__ == '__main__':
    pass
