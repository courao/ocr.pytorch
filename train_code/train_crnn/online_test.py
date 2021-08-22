import torch
import utils
import mydataset
from PIL import Image
import numpy as np
import crnn as crnn
import cv2
import torch.nn.functional as F
import keys
import config

alphabet = keys.alphabet_v2
converter = utils.strLabelConverter(alphabet.copy())


def val(net, dataset, criterion, max_iter=100):
    print("Start val")
    for p in net.parameters():
        p.requires_grad = False

    num_correct, num_all = val_model(
        config.val_infofile,
        net,
        True,
        log_file="compare-" + config.saved_model_prefix + ".log",
    )
    accuracy = num_correct / num_all

    print("ocr_acc: %f" % (accuracy))
    global best_acc
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(
            crnn.state_dict(),
            "{}_{}_{}.pth".format(
                config.saved_model_prefix,
                epoch,
                int(best_acc * 1000),
            ),
        )
    torch.save(crnn.state_dict(), f"{config.saved_model_prefix}")


def val_model(infofile, model, gpu, log_file="0625.log"):
    h = open("log/{}".format(log_file), "w")
    with open(infofile) as f:
        content = f.readlines()
        num_all = 0
        num_correct = 0

        for line in content:
            if "\t" in line:
                fname, label = line.split("\t")
            else:
                fname, label = line.split("g:")
                fname += "g"
            label = label.replace("\r", "").replace("\n", "")
            img = cv2.imread(fname)
            res = val_on_image(img, model, gpu)
            res = res.strip()
            label = label.strip()
            if res == label:
                num_correct += 1
            else:
                print("filename:{}\npred  :{}\ntarget:{}".format(fname, res, label))
                h.write("filename:{}\npred  :{}\ntarget:{}\n".format(fname, res, label))
            num_all += 1
    h.write(
        "ocr_correct: {}/{}/{}\n".format(num_correct, num_all, num_correct / num_all)
    )
    print(num_correct / num_all)
    h.close()
    return num_correct, num_all


def val_on_image(img, model, gpu):
    imgH = config.imgH
    h, w = img.shape[:2]
    imgW = imgH * w // h

    transformer = mydataset.resizeNormalize((imgW, imgH), is_test=True)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(np.uint8(img)).convert("L")
    image = transformer(image)
    if gpu:
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    preds = F.log_softmax(preds, 2)
    conf, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    # raw_pred = converter.decode( preds.data, preds_size.data, raw=True )
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred


if __name__ == "__main__":
    pass
