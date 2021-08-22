import keys

alphabet = keys.alphabet_v2
imgH = 32
imgW = 800
nc = 1
nclass = len(alphabet) + 1
nh = 256
niter = 100
lr = 0.0003
beta1 = 0.5
cuda = True
ngpu = 1
saved_model_dir = "crnn_models"
remove_blank = False


saved_model_prefix = "CRNN-1010"
train_infofile = ["path_to_train_infofile1.txt", "path_to_train_infofile2.txt"]
val_infofile = "path_to_test_infofile.txt"
keep_ratio = True
use_log = True
pretrained_model = "path_to_your_pretrained_model.pth"
batchSize = 80
workers = 0  # make 10 if gpu
adam = True


experiment = None
displayInterval = 500
n_test_disp = 10
valInterval = 500
saveInterval = 500
adadelta = False
random_sample = True


max_epochs = 30
