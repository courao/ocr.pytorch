import torch
from torch.autograd import Variable
import utils
import mydataset
from PIL import Image
import numpy as np
import crnn as crnn
import cv2
import torch.nn.functional as F
import keys
import config
gpu = True
if not torch.cuda.is_available():
    gpu = False

model_path = './crnn_models/CRNN-0618-10w_21_990.pth'
alphabet = keys.alphabet
print(len(alphabet))
imgH = config.imgH
imgW = config.imgW
model = crnn.CRNN(imgH, 1, len(alphabet) + 1, 256)
if gpu:
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
if gpu:
    model.load_state_dict( torch.load( model_path ) )
else:
    model.load_state_dict(torch.load(model_path,map_location=lambda storage,loc:storage))

converter = utils.strLabelConverter(alphabet)
transformer = mydataset.resizeNormalize((imgW, imgH),is_test=True)

def recognize_downline(img,crnn_model=model):
    img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
    image = Image.fromarray(np.uint8(img)).convert('L')
    image = transformer( image )
    if gpu:
        image = image.cuda()
    image = image.view( 1, *image.size() )
    image = Variable( image )

    model.eval()
    preds = model( image )

    preds = F.log_softmax(preds,2)
    conf, preds = preds.max( 2 )
    preds = preds.transpose( 1, 0 ).contiguous().view( -1 )

    preds_size = Variable( torch.IntTensor( [preds.size( 0 )] ) )
    raw_pred = converter.decode( preds.data, preds_size.data, raw=True )
    sim_pred = converter.decode( preds.data, preds_size.data, raw=False )
    return sim_pred.upper()


if __name__ == '__main__':
    import shutil
    saved_path = 'test_imgs/'
    wrong_results = list()
    with open('data_set/infofile_test.txt') as f:
        content = f.readlines()
        num_all = 0
        num_correct = 0
        for line in content:
            fname, label = line.split('g:')
            fname += 'g'
            label = label.replace('\r', '').replace('\n', '')
            img = cv2.imread(fname)
            res = recognize_downline(img)
            if res==label:
                num_correct+=1
            else:
                # new_name = saved_path + fname.split('/')[-1]
                # shutil.copyfile(fname, new_name)
                wrong_results.append('res:{} / label:{}'.format(res,label))
            num_all+=1
            print(fname,res==label,res,label)

        print(num_correct/num_all)
        # print(wrong_results)





