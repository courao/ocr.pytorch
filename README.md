## ocr.pytorch
> A pure pytorch implemented ocr project.  
Still under developing!  
Only test codes are available now, training codes will come soon. 
### Detection
Detection is based on [CTPN](https://arxiv.org/abs/1609.03605), some codes borrowed from 
[pytorch_ctpn](https://github.com/opconty/pytorch_ctpn), several detection results: 
![detect1](test_result/t1.png)
![detect2](test_result/t2.png)
### Recognition
Recognition is based on [CRNN](http://arxiv.org/abs/1507.05717), some codes borrowed from
[crnn.pytorch](https://github.com/meijieru/crnn.pytorch)

### Test
Download pretrained models from [here](https://pan.baidu.com/s/1yllO9hBF8TgChHJ7i3WobA) (extract code: u2ff)
and put these files into checkpoints.
Then run
>python3 demo.py

The image files in ./test_images will be tested for text detection and recognition, the results will be stored in ./test_result.

If you want to test a single image, run
>python3 test_one.py [filename]

### Train
TBC

### Licence
[MIT License](https://opensource.org/licenses/MIT)