#-*- coding:utf-8 -*-
#'''
# Created on 18-12-11 上午10:03
#
# @Author: Greg Gao(laygin)
#'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from ctpn_model import CTPN_Model
from ctpn_utils import gen_anchor, bbox_transfor_inv, clip_box, filter_bbox,nms, TextProposalConnectorOriented
from ctpn_utils import resize
import config


prob_thresh = 0.5
width = 960
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
weights = os.path.join(config.checkpoints_dir, 'v3_ctpn_ep30_0.3699_0.0929_0.4628.pth')#'ctpn_ep17_0.0544_0.1125_0.1669.pth')


model = CTPN_Model()
model.load_state_dict(torch.load(weights, map_location=device)['model_state_dict'])
model.to(device)
model.eval()


def dis(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_det_boxes(image,display = True):
    image = resize(image, height=720)
    image_c = image.copy()
    h, w = image.shape[:2]
    image = image.astype(np.float32) - config.IMAGE_MEAN
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()

    with torch.no_grad():
        image = image.to(device)
        cls, regr = model(image)
        cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
        regr = regr.cpu().numpy()
        anchor = gen_anchor((int(h / 16), int(w / 16)), 16)
        bbox = bbox_transfor_inv(anchor, regr)
        bbox = clip_box(bbox, [h, w])
        # print(bbox.shape)

        fg = np.where(cls_prob[0, :, 1] > prob_thresh)[0]
        # print(np.max(cls_prob[0, :, 1]))
        select_anchor = bbox[fg, :]
        select_score = cls_prob[0, fg, 1]
        select_anchor = select_anchor.astype(np.int32)
        # print(select_anchor.shape)
        keep_index = filter_bbox(select_anchor, 16)

        # nms
        select_anchor = select_anchor[keep_index]
        select_score = select_score[keep_index]
        select_score = np.reshape(select_score, (select_score.shape[0], 1))
        nmsbox = np.hstack((select_anchor, select_score))
        keep = nms(nmsbox, 0.3)
        # print(keep)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]

        # text line-
        textConn = TextProposalConnectorOriented()
        text = textConn.get_text_lines(select_anchor, select_score, [h, w])
        print(text)
        if display:
            for i in text:
                s = str(round(i[-1] * 100, 2)) + '%'
                i = [int(j) for j in i]
                cv2.line(image_c, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)
                cv2.line(image_c, (i[0], i[1]), (i[4], i[5]), (0, 0, 255), 2)
                cv2.line(image_c, (i[6], i[7]), (i[2], i[3]), (0, 0, 255), 2)
                cv2.line(image_c, (i[4], i[5]), (i[6], i[7]), (0, 0, 255), 2)
                cv2.putText(image_c, s, (i[0]+13, i[1]+13),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255,0,0),
                            2,
                            cv2.LINE_AA)

        return text,image_c

if __name__ == '__main__':
    img_path = 'images/t1.png'
    image = cv2.imread(img_path)
    text,image = get_det_boxes(image)
    cv2.imwrite('results/t.jpg',image)
    # dis(image)