import torch
from torch.autograd import Variable
import crnn_utils
import mydataset
from PIL import Image
import numpy as np
import crnn_model as crnn
import cv2
import torch.nn.functional as F
import keys

gpu = True
if not torch.cuda.is_available():
    gpu = False

model_path = './checkpoints/CRNN.pth'
alphabet = keys.alphabet
imgH = 32
imgW = 280
model = crnn.CRNN(imgH, 1, len(alphabet) + 1, 256)
if gpu:
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
if gpu:
    model.load_state_dict( torch.load( model_path ) )
else:
    model.load_state_dict(torch.load(model_path,map_location=lambda storage,loc:storage))
model.eval()
print('done')
print('starting...')
converter = crnn_utils.strLabelConverter(alphabet)
transformer = mydataset.resizeNormalize3((imgW, imgH))

def recognize_cv2_image(img):
    img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
    image = Image.fromarray(np.uint8(img)).convert('L')
    image = transformer( image )
    if gpu:
        image = image.cuda()

    preds = model( image )
    preds = F.log_softmax(preds,2)
    conf, preds = preds.max( 2 )
    preds = preds.transpose( 1, 0 ).contiguous().view( -1 )

    preds_size = Variable( torch.IntTensor( [preds.size( 0 )] ) )
    raw_pred = converter.decode( preds.data, preds_size.data, raw=True )
    sim_pred = converter.decode( preds.data, preds_size.data, raw=False )
    return sim_pred.upper()

def recognize_PIL_image(img):
    image = transformer(img)
    if gpu:
        image = image.cuda()

    preds = model(image)
    preds = F.log_softmax(preds, 2)
    conf, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred.upper()

if __name__ == '__main__':
    fname = 'data/images/000059.jpg'
    img = cv2.imread(fname)
    res = recognize_cv2_image(img)
    print(res)





