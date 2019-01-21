from __future__ import print_function

import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, \
    Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval
import csv_eval

import commands

from tensorboardX import SummaryWriter

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

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

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--pretrained', help='Path to file containing pretrained model on COCO dataset (optional, see readme)')
    parser.add_argument('--model_name', help='Name of the model being saved to drive (optional, see readme)')
    parser.add_argument('--project_name', help='Name of the project being saved to drive (optional, see readme)')
    # parser.add_argument('--snapshot', help='Path to file containing snapshot model (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=1)
    parser.add_argument('--images_period', help='Period for representing images', type=int, default=1000)
    parser.add_argument('--freeze_backbone', help='Freeze backbone', action='store_true')

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    print("The number of classes is: " + str(dataset_train.num_classes()))

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True, freeze_backbone=parser.freeze_backbone)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True, freeze_backbone=parser.freeze_backbone)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True, freeze_backbone=parser.freeze_backbone)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True, freeze_backbone=parser.freeze_backbone)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True, freeze_backbone=parser.freeze_backbone)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    if parser.pretrained is not None:
        if parser.depth == 50:
            # Load model to be trained
            retinanet_dict = retinanet.state_dict()

            # Load pretrained model
            COCO_NUM_CLASSES = 80
            retinanet_coco = model.resnet50(num_classes=COCO_NUM_CLASSES, )
            retinanet_coco.load_state_dict(torch.load(parser.pretrained))
            retinanet_coco_dict = retinanet_coco.state_dict()

            # 1. filter out unnecessary keys
            print("Retinanet state dict pre-filter length: " + str(len(retinanet_coco_dict)))
            retinanet_coco_dict = { k:v for k,v in retinanet_coco_dict.iteritems() if k in retinanet_dict and v.size() == retinanet_dict[k].size() }
            print("Retinanet state dict post-filter length: " + str(len(retinanet_coco_dict)))

            # 2. overwrite entries in the existing state dict
            retinanet_dict.update(retinanet_coco_dict)

            # 3. load the new state dict
            retinanet.load_state_dict(retinanet_dict)
        else:
            raise ValueError('Unsupported pretrained model depth, must be 50')

    # if parser.snapshot is not None:
    #     if parser.depth == 50:
    #         ## If state dicts are saved with Data Parallel and module is included
    #         # original saved file with DataParallel
    #         state_dict = torch.load(parser.snapshot)
    #         # create new OrderedDict that does not contain `module.`
    #         from collections import OrderedDict
    #         new_state_dict = OrderedDict()
    #         for k, v in state_dict.items():
    #             name = k[7:]  # remove `module.`
    #             new_state_dict[name] = v
    #         # load params
    #         retinanet.load_state_dict(new_state_dict)
    #         # Print some info
    #         print("Snapshot successfully loaded")
    #
    #     else:
    #         raise ValueError('Unsupported snapshot model depth, must be 50')


    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    if use_gpu:
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, retinanet.parameters()), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    writer = SummaryWriter('experiments')

    unnormalize = UnNormalizer()

    global_step = 0

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        # Saving model
        print("Saving model at epoch: " + str(epoch_num))
        torch.save(retinanet.module, 'snapshots/{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))
        torch.save(retinanet.state_dict(), 'snapshots/{}_retinanet_state_dict_{}.pt'.format(parser.dataset, epoch_num))

        if parser.model_name is not None:
            # Saving model back to drive
            print("Saving back to Drive.. model(" + parser.model_name + ")")
            drive_path = 'cp -rf snapshots/*' + ' /content/drive/My Drive/PhD/cloud/projects/' + parser.project_name + '/results/training/' + parser.model_name
            commands.getstatusoutput(drive_path)

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if use_gpu:
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.001)

                optimizer.step()

                global_step += 1

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                # Each 1000 iterations show image
                if global_step % parser.images_period == 0:
                    st = time.time()
                    retinanet.eval()
                    scores, classification, transformed_anchors = retinanet(data['img'].float().cuda())
                    retinanet.train()
                    print('Elapsed time: {}'.format(time.time() - st))
                    idxs = np.where(scores.cpu().detach().numpy() > 0.5)

                    print(data['img'].squeeze().size())

                    # img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
                    #
                    # img[img < 0] = 0
                    # img[img > 255] = 255

                    # img = np.transpose(img, (1, 2, 0))

                    img_tensor = unnormalize(data['img']).squeeze().float().cpu()
                    img_tensor[img_tensor < 0] = 0
                    img_tensor[img_tensor > 1] = 1

                    detected_object = False

                    if transformed_anchors.size()[0] != 0:
                        # Apply non-max suprresion
                        boxes = non_max_suppression_fast(transformed_anchors.cpu().detach().numpy(), 0.5)
                        # print("Boxes shape: " + str(boxes.shape))

                        bbox_tensor = torch.randn(boxes.shape[0], 4, dtype=torch.float)

                        for j in range(boxes.shape[0]):
                            bbox = boxes[j, :]
                            # print("Transformed anchors shape: " + str(transformed_anchors.shape))
                            # print("idxs shape: " + str(idxs[0].shape))
                            # bbox = transformed_anchors[idxs[0][j], :]

                            bbox_tensor[j, 0] = float(bbox[0])
                            bbox_tensor[j, 1] = float(bbox[1])
                            bbox_tensor[j, 2] = float(bbox[2])
                            bbox_tensor[j, 3] = float(bbox[3])

                            detected_object = True

                            # Enconding the class number (classification result) into the represented global step
                            # tmp_step = global_step + idxs[0][0]

                        writer.add_image_with_boxes("Image eval", img_tensor, bbox_tensor, global_step=global_step)
                        print("Detection of object in image (classes: " + str(classification.cpu()) + ")")
                        print("with scores: " + str(scores.cpu()))

                    if not detected_object:
                        writer.add_image("Image eval", img_tensor, global_step=global_step)
                        print("No detected object")

                # print(
                #     '\r Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                #         epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                # Plot in tensorboard
                writer.add_scalar('Classification loss', float(classification_loss), global_step)
                writer.add_scalar('Regression loss', float(regression_loss), global_step)
                writer.add_scalar('Running loss', np.mean(loss_hist), global_step)

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        writer.add_scalar('Epoch loss', np.mean(loss_hist), epoch_num)

    retinanet.eval()

    torch.save(retinanet, 'model_final.pt'.format(epoch_num))

    writer.close()


if __name__ == '__main__':
    main()
