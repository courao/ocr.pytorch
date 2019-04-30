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

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output

class CRNN_vgg(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN_vgg, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def conv_relu_batchnorm(i):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module( 'conv{0}'.format( i ),
                            nn.Conv2d( nIn, nOut, ks[i], ss[i], ps[i] ) )
            cnn.add_module( 'batchnorm{0}'.format( i ), nn.BatchNorm2d( nOut ) )
            if leakyRelu:
                cnn.add_module( 'relu{0}'.format( i ),
                                nn.LeakyReLU( 0.2, inplace=True ) )
            else:
                cnn.add_module( 'relu{0}'.format( i ), nn.ReLU( True ) )


        conv_relu_batchnorm(0)  # 64x32x128
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        conv_relu_batchnorm(1)  # 128x16x64
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        conv_relu_batchnorm(2)
        conv_relu_batchnorm(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        conv_relu_batchnorm(4)
        conv_relu_batchnorm(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        conv_relu_batchnorm(6)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
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

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN_res, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def conv_res(i):
            nIn = nm[i]
            nOut = nIn
            cnn.add_module( 'res{0}'.format( i ),basic_res_block(nIn, nOut) )

        def conv_res_downsample(i,stride=(2,2)):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            downsample = nn.Sequential(
                nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(nOut),
            )
            cnn.add_module( 'res-ds{0}'.format( i ), basic_res_block( nIn, nOut, stride, downsample ) )


        def conv_relu_batchnorm(i):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module( 'conv{0}'.format( i ),
                            nn.Conv2d( nIn, nOut, ks[i], ss[i], ps[i] ) )
            cnn.add_module( 'batchnorm{0}'.format( i ), nn.BatchNorm2d( nOut ) )
            if leakyRelu:
                cnn.add_module( 'relu{0}'.format( i ),
                                nn.LeakyReLU( 0.2, inplace=True ) )
            else:
                cnn.add_module( 'relu{0}'.format( i ), nn.ReLU( True ) )

        conv_relu_batchnorm(0)  # 64x32x128
        conv_res(0)             # 64x32x128
        conv_res_downsample(1)  # 128x16x64
        conv_res(1)             # 128x16x64
        conv_res_downsample( 2 )  # 256x8x32
        conv_res(2)             # 256x8x32
        conv_res(3)             # 256x8x32
        conv_res_downsample( 4,stride=(2, 1) )  # 512x4x16
        conv_res(4)             # 512x4x16
        conv_res(5)             # 512x4x16
        cnn.add_module('pooling{0}'.format(3),
                       nn.AvgPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        conv_relu_batchnorm(6)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output