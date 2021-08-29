import cv2
from math import *
import numpy as np
from train_code.train_ctpn.ctpn_model_PL import CTPN_Model
from train_code.train_crnn.crnn_recognizer import PytorchOcr
import numpy as np


# load model once
ctpn_model = CTPN_Model()
ctpn_model.load_checkpoint()

recognizer = PytorchOcr()


def dis(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)


def sort_box(box):
    """
    Sort boxes
    """
    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box


def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4) -> np.array:
    """
    turn an image by a number of degrees
    return image as numpy array
    """
    height, width = img.shape[:2]
    heightNew = int(
        width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree)))
    )
    widthNew = int(
        height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree)))
    )
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    imgRotation = cv2.warpAffine(
        img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255)
    )
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[
        max(1, int(pt1[1])) : min(ydim - 1, int(pt3[1])),
        max(1, int(pt1[0])) : min(xdim - 1, int(pt3[0])),
    ]

    return imgOut


def charRec(img, text_recs, adjust=False) -> dict:
    """
    Chop img into text_recs (rectangles) that have text in them
    Use CRNN model for character recognition on those rectangles

    Returns dict of:
    {
        1: [<bbox>, <text>]
        2: [<bbox>, <text>]
        ...
    }
    """
    results = {}
    xDim, yDim = img.shape[1], img.shape[0]

    # rec: large rectangles with text inside
    for index, rec in enumerate(text_recs):
        xlength = int((rec[6] - rec[0]) * 0.1)
        ylength = int((rec[7] - rec[1]) * 0.2)
        if adjust:
            pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
            pt4 = (rec[4], rec[5])
        else:
            pt1 = (max(1, rec[0]), max(1, rec[1]))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
            pt4 = (rec[4], rec[5])

        # tilt image if rectangle is slanted
        # we want straight rectangles to go into CRNN
        degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))
        partImg = dumpRotateImage(img, degree, pt1, pt2, pt3, pt4)

        # Filter out images with x, y dimensions == 0
        if (
            partImg.shape[0] < 1  # x dimension == 0
            or partImg.shape[1] < 1  # y dimension == 0
            or partImg.shape[0] > partImg.shape[1]  # x-dim > y-dim
        ):
            continue

        # Recognize text on those tiny boxes
        text = recognizer.recognize(partImg)

        if len(text) > 0:  # make sure text != ""
            results[index] = [rec] + [text]

    return results


def ocr(image: np.array):
    """
    Detection of text in 3 steps
    1) use CTPN to detect large boxes of text
    2) sort large boxes, converting to something CRNN would understand
    3) use CRNN to detect text in those boxes
    4) return images with text
    """
    # detect large boxes of text (CTPN)
    text_recs, img_framed, image = ctpn_model.get_det_boxes(image)
    # sort large boxes, converting to something CRNN would understand
    text_recs = sort_box(text_recs)
    # detect characters on those large boxes (CRNN)
    result = charRec(image, text_recs)

    return result, img_framed
