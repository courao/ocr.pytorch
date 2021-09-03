## Train CTPN

> Modified codes from [pytorch_ctpn](https://github.com/opconty/pytorch_ctpn)  

To train your own model, put your images into one directory [images], 
and labels into another directory [labels].  
Replace the value of icdar17_mlt_img_dir and icdar17_mlt_gt_dir in config.py by your own path.  
Then run

>python3 ctpn_train.py  

If you want to train on ICDAR datasets, please visit [here](https://rrc.cvc.uab.es) and download datasets you like.


Data comes in the form of:
x1,y1,x2,y2,x3,y3,x4,y4,script,transcription
