import os
import csv
import torch
import logging
import time
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *
from util import Logger, printSet
import numpy as np

logging.basicConfig(filename='log_test.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

# Running tests
opt = TestOptions().parse(print_options=False)
model_name = os.path.basename(model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision']]

log_msg = f'Model_path {opt.model_path}'
print(log_msg)
logger.info(log_msg)
print("{} model testing on...".format(model_name))
accs = []
aps = []
r_accs = []
f_accs = []
current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(current_time)
logger.info(current_time)
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)
    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    opt.no_resize = True    # testing without resizing by default

    model = resnet50(num_classes=1)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, ap, r_acc, f_acc, _, _ = validate(model, opt)
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