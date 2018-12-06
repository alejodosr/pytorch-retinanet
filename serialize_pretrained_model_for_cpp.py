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

import model_cpp
# from anchors import Anchors
# import losses
# from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
# from torch.utils.data import Dataset, DataLoader
#
# import coco_eval
# import csv_eval

# normalize = transforms.Normalize(
#    mean=[0.485, 0.456, 0.406],
#    std=[0.229, 0.224, 0.225]
# )
# preprocess = transforms.Compose([
#    transforms.Resize(256),
#    transforms.CenterCrop(224),
#    transforms.ToTensor()
#    # normalize
#    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# resnet_model = torchvision.models.resnet50(pretrained=True, num_classes=1000)
# resnet_model.eval()
#
# for i in range(4):
#     # An instance of your model.
#     img_pil = Image.open("/home/alejandro/workspace/uav_detection/images/" + str(i + 1) + ".jpg")
#     # img_pil.show()
#     img_tensor = preprocess(img_pil).float()
#     img_tensor = img_tensor.unsqueeze_(0)
#
#     fc_out = resnet_model(Variable(img_tensor))
#
#     output = fc_out.detach().numpy()
#     print(output.argmax())

resnet_model = torchvision.models.resnet50(pretrained=True, num_classes=1000)
resnet_model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(resnet_model, example)

output = traced_script_module(torch.ones(1, 3, 224, 224))
print(output)
print("Size of the output: " + str(output.size()))
traced_script_module.save("resnet18_imagenet.pt")
