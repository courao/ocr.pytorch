## lightning_text_detection
> Applying pytorch-lightning to CTPN and CRNN
> Code heavily borrowed from [courao](https://github.com/courao/ocr.pytorch)

Text detection is based CTPN and text recognition is based CRNN.  
More detection and recognition methods will be supported!


Working on implementing [CRAFT](https://github.com/clovaai/CRAFT-pytorch)
And [Transformer_STR](https://github.com/opconty/Transformer_STR)

Pull requests welcome!

## Prerequisite

- python-3.6+
- pytorch-lightning-1.4.1
- pytorch-1.8.1
- torchvision-0.9.1
- opencv-4.5.2.52
- numpy-1.21.1
- Pillow-8.2.0
- Paths-0.0.34


### Detection
Detection is based on [CTPN](https://arxiv.org/abs/1609.03605), some codes are borrowed from 
[pytorch_ctpn](https://github.com/opconty/pytorch_ctpn)

### Recognition
Recognition is based on [CRNN](http://arxiv.org/abs/1507.05717), some codes are borrowed from
[crnn.pytorch](https://github.com/meijieru/crnn.pytorch)

### Test
Download pretrained models from [Baidu Netdisk](https://pan.baidu.com/s/1yllO9hBF8TgChHJ7i3WobA) (extract code: u2ff) or [Google Driver](https://drive.google.com/open?id=1hRr9v9ky4VGygToFjLD9Cd-9xan43qID)
and put these files into checkpoints.
Then run
>python3 demo.py

The image files in ./test_images will be tested for text detection and recognition, the results will be stored in ./test_result.

If you want to test a single image, run
>python3 test_one.py [filename]

### Train
Training codes are placed into train_code directory.  
Train [CTPN](./train_code/train_ctpn/readme.md)  
Train [CRNN](./train_code/train_crnn/readme.md)  

### Licence
[MIT License](https://opensource.org/licenses/MIT)