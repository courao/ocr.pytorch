from __future__ import print_function
import argparse
import random
import torch
# import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
# from torch.nn import CTCLoss
import utils
import mydataset
import crnn as crnn
import config
from online_test import val_model
config.imgW = 800
config.alphabet = config.alphabet_v2
config.nclass = len(config.alphabet) + 1
config.saved_model_prefix = 'CRNN-1010'
config.train_infofile = ['path_to_train_infofile1.txt','path_to_train_infofile2.txt']
config.val_infofile = 'path_to_test_infofile.txt'
config.keep_ratio = True
config.use_log = True
config.pretrained_model = 'path_to_your_pretrained_model.pth'
config.batchSize = 80
config.workers = 10
config.adam = True
# config.lr = 0.00003
import os
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
log_filename = os.path.join('log/','loss_acc-'+config.saved_model_prefix + '.log')
if not os.path.exists('debug_files'):
    os.mkdir('debug_files')
if not os.path.exists(config.saved_model_dir):
    os.mkdir(config.saved_model_dir)
if config.use_log and not os.path.exists('log'):
    os.mkdir('log')
if config.use_log and os.path.exists(log_filename):
    os.remove(log_filename)
if config.experiment is None:
    config.experiment = 'expr'
if not os.path.exists(config.experiment):
    os.mkdir(config.experiment)

config.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", config.manualSeed)
random.seed(config.manualSeed)
np.random.seed(config.manualSeed)
torch.manual_seed(config.manualSeed)

# cudnn.benchmark = True
train_dataset = mydataset.MyDataset(info_filename=config.train_infofile)
assert train_dataset
if not config.random_sample:
    sampler = mydataset.randomSequentialSampler(train_dataset, config.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(config.workers),
    collate_fn=mydataset.alignCollate(imgH=config.imgH, imgW=config.imgW, keep_ratio=config.keep_ratio))

test_dataset = mydataset.MyDataset(
    info_filename=config.val_infofile, transform=mydataset.resizeNormalize((config.imgW, config.imgH), is_test=True))

converter = utils.strLabelConverter(config.alphabet)
# criterion = CTCLoss(reduction='sum',zero_infinity=True)
criterion = CTCLoss()
best_acc = 0.9


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(config.imgH, config.nc, config.nclass, config.nh)
if config.pretrained_model!='' and os.path.exists(config.pretrained_model):
    print('loading pretrained model from %s' % config.pretrained_model)
    crnn.load_state_dict(torch.load(config.pretrained_model))
else:
    crnn.apply(weights_init)

print(crnn)

# image = torch.FloatTensor(config.batchSize, 3, config.imgH, config.imgH)
# text = torch.IntTensor(config.batchSize * 5)
# length = torch.IntTensor(config.batchSize)
device = torch.device('cpu')
if config.cuda:
    crnn.cuda()
    # crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    # image = image.cuda()
    device = torch.device('cuda:0')
    criterion = criterion.cuda()

# image = Variable(image)
# text = Variable(text)
# length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if config.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
elif config.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=config.lr)
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=config.lr)


def val(net, dataset, criterion, max_iter=100):
    print('Start val')
    for p in net.parameters():
        p.requires_grad = False

    num_correct,  num_all = val_model(config.val_infofile,net,True,log_file='compare-'+config.saved_model_prefix+'.log')
    accuracy = num_correct / num_all

    print('ocr_acc: %f' % (accuracy))
    if config.use_log:
        with open(log_filename, 'a') as f:
            f.write('ocr_acc:{}\n'.format(accuracy))
    global best_acc
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(crnn.state_dict(), '{}/{}_{}_{}.pth'.format(config.saved_model_dir, config.saved_model_prefix, epoch,
                                                               int(best_acc * 1000)))
    torch.save(crnn.state_dict(), '{}/{}.pth'.format(config.saved_model_dir, config.saved_model_prefix))


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    image = cpu_images.to(device)

    text, length = converter.encode(cpu_texts)
    # utils.loadData(text, t)
    # utils.loadData(length, l)

    preds = net(image)  # seqLength x batchSize x alphabet_size
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))  # seqLength x batchSize
    cost = criterion(preds, text, preds_size, length) / batch_size
    if torch.isnan(cost):
        print(batch_size,cpu_texts)
    else:
        net.zero_grad()
        cost.backward()
        optimizer.step()
    return cost


for epoch in range(config.niter):
    loss_avg.reset()
    print('epoch {}....'.format(epoch))
    train_iter = iter(train_loader)
    i = 0
    n_batch = len(train_loader)
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()
        cost = trainBatch(crnn, criterion, optimizer)
        print('epoch: {} iter: {}/{} Train loss: {:.3f}'.format(epoch, i, n_batch, cost.item()))
        loss_avg.add(cost)
        loss_avg.add(cost)
        i += 1
    print('Train loss: %f' % (loss_avg.val()))
    if config.use_log:
        with open(log_filename, 'a') as f:
            f.write('{}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')))
            f.write('train_loss:{}\n'.format(loss_avg.val()))

    val(crnn, test_dataset, criterion)


