import keys

train_infofile = 'data_set/infofile_train_10w.txt'
train_infofile_fullimg = ''
val_infofile = 'data_set/infofile_test.txt'
alphabet = keys.alphabet
alphabet_v2 = keys.alphabet_v2
workers = 4
batchSize = 50
imgH = 32
imgW = 280
nc = 1
nclass = len(alphabet)+1
nh = 256
niter = 100
lr = 0.0003
beta1 = 0.5
cuda = True
ngpu = 1
pretrained_model = ''
saved_model_dir = 'crnn_models'
saved_model_prefix = 'CRNN-'
use_log = False
remove_blank = False

experiment = None
displayInterval = 500
n_test_disp = 10
valInterval = 500
saveInterval = 500
adam = False
adadelta = False
keep_ratio = False
random_sample = True

