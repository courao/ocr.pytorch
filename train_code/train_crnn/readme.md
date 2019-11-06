## Train CRNN

To train your own model, firstly you need to prepare your own text-line dataset, and organize your dataset with an infoile.
In the infofile, each line represents an text-line image, path to image and image label are splited by a special character '\t'.
Such as follows:
>data_set/my_data1/0001.jpg\t37918  
data_set/my_data1/0002.jpg\tHello World!  
data_set/my_data1/0003.jpg\t你好  
...

Then replace your infofile by your own in the training file train_warp_ctc.py
>config.train_infofile = ['path_to_train_infofile1.txt','path_to_train_infofile2.txt']  
config.val_infofile = 'path_to_test_infofile.txt'

Then if you want to use warp-ctc as loss function, run 
>python3 train_warp_ctc.py

or if you want to use pytorch-ctc as loss function, run
>python3 train_pytorch_ctc.py


