import sys
import time
import os
import csv
import torch
import logging
from util import Logger, printSet
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
from data import create_dataloader
import numpy as np
import random

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

seed_torch(100)

DetectionTests = {
    # 'AIGIBench': { 'dataroot': '/data/ziqiang/Benchmark',
    #              'no_resize': False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
    #              'no_crop': True,
    #            },
    'jpeg': { 'dataroot': '/data/ziqiang/jpeg',
                 'no_resize': False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                 'no_crop': True,
               },
    'noise': { 'dataroot': '/data/ziqiang/noise',
                 'no_resize': False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                 'no_crop': True,
               },
    'sample': { 'dataroot': '/data/ziqiang/sample',
                 'no_resize': False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                 'no_crop': True,
               },
}


# Set up logging
logging.basicConfig(filename='log_test.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

opt = TestOptions().parse(print_options=False)
log_msg = f'Model_path {opt.model_path}'
print(log_msg)
logger.info(log_msg)

# Get model
model = resnet50(num_classes=1)
model.load_state_dict(torch.load(opt.model_path), strict=True)
model.cuda()
model.eval()

for testSet in DetectionTests.keys():
    dataroot = DetectionTests[testSet]['dataroot']
    printSet(testSet)
    logger.info(testSet)

    accs = []
    aps = []
    r_accs = []
    f_accs = []
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(current_time)
    logger.info(current_time)

    for v_id, val in enumerate(os.listdir(dataroot)):
        opt.dataroot = '{}/{}'.format(dataroot, val)
        opt.classes = '' # os.listdir(opt.dataroot) if multiclass[v_id] else ['']
        opt.no_resize = DetectionTests[testSet]['no_resize']
        opt.no_crop = DetectionTests[testSet]['no_crop']
        dataloader = create_dataloader(opt)
        acc, ap, r_acc, f_acc, _, _ = validate(model, dataloader)
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
