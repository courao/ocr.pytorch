import os
from ocr import ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
import cv2


def single_pic_proc(image_file):
    image = cv2.imread(image_file)
    h, w, c = image.shape
    rescale_fac = max(h, w) / 1600
    if rescale_fac > 1.0:
        h = int(h / rescale_fac)
        w = int(w / rescale_fac)
        image = cv2.resize(image, (w, h))

    result, image_framed = ocr(image)
    return result, image_framed


if __name__ == "__main__":
    image_files = glob("./test_images/*.*")
    result_dir = "./test_result"
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    for image_file in sorted(image_files):
        t = time.time()
        result, image_framed = single_pic_proc(image_file)
        output_file = os.path.join(result_dir, image_file.split("/")[-1])
        txt_file = os.path.join(
            result_dir, image_file.split("/")[-1].split(".")[0] + ".txt"
        )
        print(txt_file)
        txt_f = open(txt_file, "w")
        Image.fromarray(image_framed).save(output_file)
        print("Mission complete, it took {:.3f}s".format(time.time() - t))

        print("\nRecognition Result:\n")
        for key in result:
            print(result[key][1])
            txt_f.write(result[key][1] + "\n")
        txt_f.close()
