from pathed import filedir
import pickle as pkl

alphabet_list = pkl.load(open(filedir / "alphabet.pkl", "rb"))
alphabet = [ord(ch) for ch in alphabet_list]

alphabet = alphabet
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

project_directory = filedir / ".." / ".."
saved_model_prefix = "CRNN-1010"
train_infofile = [project_directory / "ocr_pytorch/text_file.txt"]
val_infofile = "path_to_test_infofile.txt"
keep_ratio = True
use_log = True
pretrained_model = project_directory / "CRNN-1010.pth"
batchSize = 9  # make 80 if gpu
workers = 0  # make 10 if gpu
adam = True


experiment = None
displayInterval = 500
n_test_disp = 9
valInterval = 500
saveInterval = 500
adadelta = False
random_sample = True


max_epochs = 10
