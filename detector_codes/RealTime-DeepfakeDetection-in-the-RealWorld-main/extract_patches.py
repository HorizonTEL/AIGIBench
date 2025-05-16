
import os
import sys
import torch
import torch.nn
import argparse
import numpy as np
from data import create_dataloader
from networks.LaDeDa import LaDeDa9
from options.train_options import TrainOptions
import torchvision
from torchvision import transforms
import torch.nn as nn


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True

    return val_opt

def get_avg_logits(model, x):
    x = model.conv1(x)
    x = model.conv2(x)
    x = model.bn1(x)
    x = model.relu(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    if model.pool:
        x = nn.AvgPool2d(x.size()[2], stride=1)(x)
        x = x.view(x.size(0), -1)
        x = model.fc(x)
    else:
        x = x.permute(0, 2, 3, 1)
        x = model.fc(x)

    return x

def get_patches_logits(image, patchsize, teacher, preprocess):
    npr_img = model.preprocess(image, preprocess)
    teacher_patches = npr_img.permute(0, 2, 3, 1)
    teacher_patches = teacher_patches.unfold(1, patchsize, patchsize - 1).unfold(2, patchsize, patchsize - 1)
    teacher_patches = teacher_patches.contiguous().view((-1, 3, patchsize, patchsize))

    og_patches = image.permute(0, 2, 3, 1)
    og_patches = og_patches.unfold(1, patch_size + 1, patchsize - 1).unfold(2, patchsize + 1, patch_size - 1)
    og_patches = og_patches.contiguous().view((-1, 3, patchsize+1, patchsize+1))

    # getting the teacher logits
    teacher_logits = get_avg_logits(teacher, teacher_patches)

    return og_patches.detach().cpu(), teacher_logits.detach().cpu()


def get_model(model_path, features_dim):
    model = LaDeDa9(pretrained=False, num_classes=1)
    model.fc = nn.Linear(features_dim, 1)
    from collections import OrderedDict
    from copy import deepcopy
    state_dict = torch.load(model_path, map_location='cpu')
    pretrained_dict = OrderedDict()
    for ki in state_dict.keys():
        pretrained_dict[ki] = deepcopy(state_dict[ki])
    model.load_state_dict(pretrained_dict, strict=True)
    print("LaDeDa has loaded")
    model.eval()
    model.cuda()
    model.to(0)
    return model

def extract_patches_logits(data_loader, model):
    distilled_set = {"real": [], "fake": []}
    for i, data in enumerate(data_loader):
        img, label = data
        img_input = img.cuda()
        patches, logits = get_patches_logits(img_input, patchsize=9, teacer=model)
        to_save = {"patches": patches,
                   "logits": logits,
                   "label": label.item()}
        # saving the real patches
        if label.item() == 0:
            distilled_set["real"].append(to_save)

        # saving the fake patches
        elif label.item() == 1:
            distilled_set["fake"].append(to_save)
    return distilled_set

if __name__ == '__main__':
    opt = TrainOptions().parse()
    # no flipping the image, as we extract patches logits.
    opt.no_flip = True
    opt.dataroot = '{}/{}'.format(opt.dataroot, opt.train_split)
    data_loader, paths = create_dataloader(opt)

    # getting pre-trained LaDeDa teacher
    model_path = "PATH_TO_TRAINED_LADEDA.pth"
    model = get_model(model_path, features_dim=2048)
    torch.multiprocessing.set_sharing_strategy('file_system')
    distilled_train = {"real": [], "fake": []}
    for i, data in enumerate(data_loader):
        img, label = data
        img_input = img.cuda()
        patches, logits = get_patches_logits(img_input, patchsize=9, teacer=model, preprocess=opt.preprocess)
        to_save = {"patches": patches,
                   "logits": logits,
                   "label": label.item()}
        # saving the real patches
        if label.item() == 0:
            distilled_train["real"].append(to_save)

        # saving the fake patches
        elif label.item() == 1:
            distilled_train["fake"].append(to_save)


    # saving the train set
    distilled_train = extract_patches_logits(data_loader, model)
    print("======================================")
    print("saving Tiny-LaDeDa's train set")
    print("======================================")
    np.savez("./patches_logits_train_set.npz", **distilled_train)

    # saving the validation set
    val_opt = get_val_opt()
    data_loader, paths = create_dataloader(val_opt)
    distilled_val = extract_patches_logits(data_loader, model)
    print("======================================")
    print("saving Tiny-LaDeDa's validation set")
    print("======================================")
    np.savez("./patches_logits_val_set.npz", **distilled_val)
