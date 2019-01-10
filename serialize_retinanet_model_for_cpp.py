# import time
# import os
# import copy
# import argparse
# import pdb
# import collections
# import sys
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



# Test serialized
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor()
   # normalize
   # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Code for pyTorch retinanet
# PATH_TO_WEIGHTS="/home/alejandro/py_workspace/pytorch-retinanet/snapshots/coco_resnet_50_map_0_335_state_dict.pt"
# PATH_TO_WEIGHTS="/home/alejandro/py_workspace/pytorch-retinanet/snapshots/coco_resnet_50_map_0_335.pt"
# PATH_TO_WEIGHTS="/home/alejandro/py_workspace/pytorch-retinanet/snapshots/csv_retinanet_state_dict_2.pt"
PATH_TO_WEIGHTS="/home/alejandro/py_workspace/pytorch-retinanet/snapshots/csv_test2_coco_state_dict_0.pt"
# PATH_TO_WEIGHTS="/home/alejandro/py_workspace/pytorch-retinanet/snapshots/csv_retinanet_2.pt"



num_classes = 80

retinanet = model_cpp.resnet50(num_classes=num_classes,)

# ## If state dicts are saved with Data Parallel and module is included
# # original saved file with DataParallel
# state_dict = torch.load(PATH_TO_WEIGHTS)
# # create new OrderedDict that does not contain `module.`
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v
# # load params
# retinanet.load_state_dict(new_state_dict)

## Else
# load params
retinanet.load_state_dict(torch.load(PATH_TO_WEIGHTS))


retinanet = retinanet.cuda()
retinanet.eval()

for i in range(7):
    # An instance of your model.
    img_pil = Image.open("/home/alejandro/workspace/uav_detection/images/" + str(i + 1) + ".jpg")
    # img_pil.show()
    img_tensor = preprocess(img_pil).float()
    img_tensor = img_tensor.unsqueeze_(0)

    regression_classification = retinanet(Variable(img_tensor.cuda()))

    regression = regression_classification[:, :, 0:4]
    classification = regression_classification[:, :, 4:5 + num_classes]

    print(regression.size())
    print(classification.size())

    anchors = retinanet.anchors(Variable(img_tensor.cuda()))

    print(anchors.size())

    transformed_anchors = retinanet.regressBoxes(anchors, regression)
    transformed_anchors = retinanet.clipBoxes(transformed_anchors, Variable(img_tensor.cuda()))

    print(transformed_anchors.size())

    scores = torch.max(classification, dim=2, keepdim=True)[0]

    print(scores.size())

    scores_over_thresh = (scores > 0.5)[0, :, 0]

    print(scores_over_thresh.size())

    # if scores_over_thresh.sum() == 0:
        # no boxes to NMS, just return
        # return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

    classification = classification[:, scores_over_thresh, :]
    print(classification.size())

    transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
    print(transformed_anchors.size())

    if transformed_anchors.size()[1] != 0:
        print(transformed_anchors[0,0,:])

        scores = scores[:, scores_over_thresh, :]
        print(scores.size())

        idxs = np.where(scores.cpu().detach().numpy() > 0.5)
        print(idxs)

    open_cv_image = np.array(img_pil)
    open_cv_image = cv2.resize(open_cv_image, (224, 224))
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    if transformed_anchors.size()[1] != 0:
        # Apply non-max suprresion
        boxes = non_max_suppression_fast(np.squeeze(transformed_anchors.cpu().detach().numpy(), axis=0), 0.5)
        print(boxes.shape)

        for j in range(boxes.shape[0]):
            # bbox = transformed_anchors.cpu().detach().numpy()[idxs[0][j], :]
            bbox = boxes[j, :]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])

            cv2.rectangle(open_cv_image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    cv2.imshow('img', open_cv_image)
    cv2.waitKey(0)