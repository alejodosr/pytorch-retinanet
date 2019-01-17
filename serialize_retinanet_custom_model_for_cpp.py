# import time
# import os
# import copy
# import argparse
# import pdb
# import collections
# import sys

import skimage.io
import skimage.transform
import skimage.color
import skimage
#
import numpy as np

import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision
from PIL import Image
from torch.autograd import Variable
import cv2

import model_cpp

from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, \
    Normalizer

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    v, idx = scores.sort(0)  # sort in ascending order



    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()


    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view

        print(x1.size())

        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)

        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])

        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap

        # print(idx)
        # print(idx.size())
        # idx = idx[IoU.le(overlap)]
        #
        # print(idx)
        # print(idx.size())
        # print(raw_input())
    return keep, count

# from anchors import Anchors
# import losses
# from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
# from torch.utils.data import Dataset, DataLoader
#
# import coco_eval
# import csv_eval

# assert torch.__version__.split('.')[1] == '4'

# # Code for pyTorch retinanet
# PATH_TO_WEIGHTS="/home/alejandro/py_workspace/pytorch-retinanet/snapshots/coco_resnet_50_map_0_335_state_dict.pt"
#
# retinanet = model_cpp.resnet50(num_classes=80,)
# retinanet.load_state_dict(torch.load(PATH_TO_WEIGHTS))
# retinanet = retinanet.cuda()
# retinanet.eval()
#
# example = torch.cuda.FloatTensor(1, 3, 224, 224).normal_()
#
# # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
# traced_script_module = torch.jit.trace(retinanet, example)
#
# output = traced_script_module(torch.cuda.FloatTensor(1, 3, 224, 224).fill_(1))
#
# print(output)
#
# traced_script_module.save("pytorch_retinanet_model.pt")

normalize = Normalizer()

# Test serialized
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   # normalize
   # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Release memory cache
torch.cuda.empty_cache()

# Code for pyTorch retinanet
# PATH_TO_WEIGHTS="/home/alejandro/py_workspace/pytorch-retinanet/snapshots/coco_resnet_50_map_0_335_state_dict.pt"
# PATH_TO_WEIGHTS="/home/alejandro/py_workspace/pytorch-retinanet/snapshots/coco_resnet_50_map_0_335.pt"
PATH_TO_WEIGHTS="/home/alejandro/py_workspace/pytorch-retinanet/snapshots/csv_retinanet_state_dict_2.pt"
# PATH_TO_WEIGHTS="/home/alejandro/py_workspace/pytorch-retinanet/snapshots/csv_test2_coco_state_dict_0.pt"
# PATH_TO_WEIGHTS="/home/alejandro/py_workspace/pytorch-retinanet/snapshots/csv_retinanet_2.pt"


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

num_classes = 3

retinanet = model_cpp.resnet50(num_classes=num_classes,)

## If state dicts are saved with Data Parallel and module is included
# original saved file with DataParallel
state_dict = torch.load(PATH_TO_WEIGHTS)
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
retinanet.load_state_dict(new_state_dict)

# ## Else
# # load params
# retinanet.load_state_dict(torch.load(PATH_TO_WEIGHTS))


retinanet = retinanet.cpu()
retinanet.eval()
unnormalize = UnNormalizer()

#### Parameters
SQUARE_SIZE = 512
SAVE_TRACED_MODEL = False
THRESHOLD = 0.5
LABELS = ["drone", "bird", "car"]

if SAVE_TRACED_MODEL ==  True:
    # Create random tensor
    example = torch.FloatTensor(1, 3, SQUARE_SIZE, SQUARE_SIZE).normal_()

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(retinanet, example)

    output = traced_script_module(torch.FloatTensor(1, 3, SQUARE_SIZE, SQUARE_SIZE).fill_(1))

    print(output)

    traced_script_module.save("pytorch_uav_retinanet_model.pt")

    print("SUCCESSFULLY TRACED AND SAVED RETINANET MODEL")
    input()

img_i = 7

while(1):
    # # An instance of your model.
    # img_pil = Image.open("/home/alejandro/workspace/uav_detection/images/" + str(i + 1) + ".jpg")
    # # img_pil.show()
    # img_tensor = preprocess(img_pil).float()
    #
    # # mean = np.array([[[0.485, 0.456, 0.406]]])
    # # std = np.array([[[0.229, 0.224, 0.225]]])
    #
    # mean = np.array([[[0.485]], [[0.456]], [[0.406]]])
    # std = np.array([[[0.229]], [[0.224]], [[0.225]]])
    #
    # img_np = img_tensor.float().numpy()
    #
    # img_np = img_np.astype(np.float32) / 255.0
    #
    # img_np = (img_np.astype(np.float32) - mean) / std
    # img_tensor = torch.tensor(img_np)
    # img_tensor = img_tensor.unsqueeze_(0)
    #
    # print(img_tensor.size())

    try:
        img_sk = skimage.io.imread("/home/alejandro/workspace/uav_detection/images/" + str(img_i + 1) + ".jpg")
    except:
        try:
            img_sk = skimage.io.imread("/home/alejandro/workspace/uav_detection/images/" + str(img_i + 1) + ".png")
        except:
            print("ERROR: No image found")
            break

    img_i += 1

    if len(img_sk.shape) == 2:
        img_sk = skimage.color.gray2rgb(img_sk)

    img_sk = img_sk.astype(np.float32) / 255.0

    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])

    img_sk = (img_sk.astype(np.float32) - mean) / std

    min_side = SQUARE_SIZE
    max_side = SQUARE_SIZE

    rows, cols, cns = img_sk.shape

    shortest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = float(min_side) / float(shortest_side)

    if(scale > 0):
        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = float(max_side) / float(largest_side)

        print("Scale of the transformation: " + str(scale))

        # resize the image with the computed scale
        img_sk = skimage.transform.resize(img_sk, (int(round(rows*scale)), int(round((cols*scale)))), anti_aliasing=True)

        print(img_sk.shape)

    rows, cols, cns = img_sk.shape

    pad_w = SQUARE_SIZE - rows%SQUARE_SIZE
    pad_h = SQUARE_SIZE - cols%SQUARE_SIZE

    if rows%SQUARE_SIZE == 0:
        pad_w = 0
    if cols%SQUARE_SIZE == 0:
        pad_h = 0

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = img_sk.astype(np.float32)

    img_tensor = torch.from_numpy(new_image.swapaxes(0,2).swapaxes(1,2)).unsqueeze_(0)

    # print(img_tensor.size())

    regression_classification = retinanet(img_tensor.cpu().float())

    regression = regression_classification[:, :, 0:4]
    classification = regression_classification[:, :, 4:5 + num_classes]

    # print(regression.size())
    # print(classification.size())

    anchors = retinanet.anchors(Variable(img_tensor.float().cpu()))

    # print(anchors.size())

    transformed_anchors = retinanet.regressBoxes(anchors.cuda(), regression.cuda())
    transformed_anchors = retinanet.clipBoxes(transformed_anchors, Variable(img_tensor.cuda()))
    transformed_anchors_saved = transformed_anchors.detach().cpu()
    classification_saved = classification.detach().cpu()

    # print(transformed_anchors.size())
    # print(transformed_anchors)

    scores = torch.max(classification, dim=2, keepdim=True)[0]

    # print(scores.size())

    scores_over_thresh = (scores > THRESHOLD)[0, :, 0]

    # print(scores_over_thresh.size())

    # if scores_over_thresh.sum() == 0:
        # no boxes to NMS, just return
        # return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

    classification = classification[:, scores_over_thresh, :]
    # print(classification.size())

    transformed_anchors = transformed_anchors[:, scores_over_thresh, :]

    scores_saved = scores.detach().cpu()
    # print("transformed_anchors_saved.size()" + str(transformed_anchors_saved.squeeze(0).size()))
    # print("scores.size()" + str(scores_saved.squeeze(0).squeeze(1).size()))

    img = np.array(255 * unnormalize(img_tensor[0, :, :, :].cpu())).copy()

    img[img < 0] = 0
    img[img > 255] = 255

    img = np.transpose(img, (1, 2, 0))

    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

    open_cv_image = img.copy()

    ##### Francisco Massa NMS #####
    keep, count = nms(transformed_anchors_saved.squeeze(0), scores_saved.squeeze(0).squeeze(1), overlap=0.5, top_k=200)
    # print(keep.size())
    # print(keep)
    # print(count)
    for i in range(count):
        # print(transformed_anchors_saved.squeeze(0)[keep[i], :])
        # print(scores_saved.squeeze(0).squeeze(1)[keep[i]])
        if float(scores_saved.squeeze(0).squeeze(1)[keep[i]]) > THRESHOLD:
            # print(classification_saved.squeeze(0).squeeze(1)[keep[i]])
            x1 = int(transformed_anchors_saved.squeeze(0)[keep[i], 0])
            y1 = int(transformed_anchors_saved.squeeze(0)[keep[i], 1])
            x2 = int(transformed_anchors_saved.squeeze(0)[keep[i], 2])
            y2 = int(transformed_anchors_saved.squeeze(0)[keep[i], 3])
            # print(classification_saved.squeeze(0).squeeze(1)[keep[i]].size())
            values, indices = torch.max(classification_saved.squeeze(0).squeeze(1)[keep[i]], dim=0, keepdim=True)
            # print("int(float(indices)): " + str(int(float(indices))))
            print(LABELS[int(float(indices))])
            label_name = LABELS[int(float(indices))]
            draw_caption(open_cv_image, (x1, y1, x2, y2), label_name)
            cv2.rectangle(open_cv_image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    cv2.imshow('img', open_cv_image)
    cmd = cv2.waitKey(0)

    if cmd==27:    # Esc key to stop
        break

    ######

    ##### Other NMS #####

    # print("Classification scores: " + str(classification))
    #
    # if transformed_anchors.size()[1] != 0:
    #     # Apply non-max suprresion
    #     boxes = non_max_suppression_fast(np.squeeze(transformed_anchors.cpu().detach().numpy(), axis=0), 0.5)
    #
    #     # for k in range(boxes.shape[0]):
    #
    #     print(boxes.shape)
    #
    #     for j in range(boxes.shape[0]):
    #         # bbox = transformed_anchors.cpu().detach().numpy()[idxs[0][j], :]
    #         bbox = boxes[j, :]
    #         x1 = int(bbox[0])
    #         y1 = int(bbox[1])
    #         x2 = int(bbox[2])
    #         y2 = int(bbox[3])
    #         label_name = "test"
    #         draw_caption(img, (x1, y1, x2, y2), label_name)
    #         cv2.rectangle(open_cv_image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
    #
    # cv2.imshow('img', open_cv_image)
    # cmd = cv2.waitKey(0)
    #
    # if cmd==27:    # Esc key to stop
    #     break

    ####