import os
import sys
import torch
import torch.nn
import argparse
import numpy as np
from options.test_options import TestOptions
from util import Logger
from validate import validate
import torchvision
from torchvision import transforms
import random
from networks.LaDeDa import LaDeDa9
import logging
from networks.Tiny_LaDeDa import tiny_ladeda
from test_config import *
import time

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"seed: {seed}")

def test_model(model):
    logging.basicConfig(filename='log_test.log', level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger()

    opt = TestOptions().parse(print_options=False)
    log_msg = f'Model_path {opt.model_path}'
    print(log_msg)
    logger.info(log_msg)

    accs, aps, r_accs, f_accs = [], [], [], []
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(current_time)
    logger.info(current_time)
    for v_id, val in enumerate(vals):
        print(f"eval on {val}")
        Testopt.dataroot = '{}/{}'.format(Testdataroot, val)
        Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else ['']
        Testopt.no_resize = False
        Testopt.no_crop = True
        Testopt.is_aug = False
        acc, ap, r_acc, f_acc = validate(model, Testopt)
        accs.append(acc)
        aps.append(ap)
        r_accs.append(r_acc)
        f_accs.append(f_acc)
        log_msg = "({} {:12}) acc: {:.1f}; ap: {:.1f}; r_acc: {:.1f}; f_acc: {:.1f}" \
            .format(v_id, val, acc * 100, ap * 100, r_acc * 100, f_acc * 100)
        print(log_msg)
        logger.info(log_msg)

    mean_acc = np.array(accs).mean() * 100
    mean_ap = np.array(aps).mean() * 100
    mean_r_acc = np.array(r_accs).mean() * 100
    mean_f_acc = np.array(f_accs).mean() * 100
    log_msg = "({} {:10}) acc: {:.1f}; ap: {:.1f}; r_acc: {:.1f}; f_acc: {:.1f}" \
        .format(v_id + 1, 'Mean', mean_acc, mean_ap, mean_r_acc, mean_f_acc)
    print(log_msg)
    logger.info(log_msg)
    print('*' * 25)
    logger.info('*' * 25)


def get_model(model_path, features_dim):
    model = LaDeDa9(num_classes=1)
    model.fc = torch.nn.Linear(features_dim, 1)
    from collections import OrderedDict
    from copy import deepcopy
    state_dict = torch.load(model_path, map_location='cpu')
    pretrained_dict = OrderedDict()
    for ki in state_dict.keys():
        pretrained_dict[ki] = deepcopy(state_dict[ki])
    model.load_state_dict(pretrained_dict, strict=True)
    print("model has loaded")
    model.eval()
    model.cuda()
    model.to(1)
    return model

if __name__ == '__main__':
    set_seed(42)
    Testopt = TestOptions().parse(print_options=False)
    # evaluate model
    # LaDeDa's features_dim = 2048
    # Tiny-LaDeDa's features_dim = 8
    model = get_model(model_path, features_dim=Testopt.features_dim)
    test_model(model)
