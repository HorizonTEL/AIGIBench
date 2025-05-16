import os
import sys
import time
import torch
import torch.nn
import argparse
from tensorboardX import SummaryWriter
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger
import torchvision
from earlystop import EarlyStopping
import random
import logging



import random


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"seed: {seed}")


# use when training without aug
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True

    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    set_seed()
    Testdataroot = os.path.join(opt.dataroot, 'test')
    opt.dataroot = '{}/{}'.format(opt.dataroot, opt.train_split)
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('  '.join(list(sys.argv)))
    val_opt = get_val_opt()
    Testopt = TestOptions().parse(print_options=False)
    train_data_loader, train_paths = create_dataloader(opt)
    dataset_size = len(train_data_loader)
    print('#training images = %d' % dataset_size)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    # initializing the model
    model = Trainer(opt)
    net_params = sum(map(lambda x: x.numel(), model.parameters()))
    print(f'Model parameters {net_params:,d}')

    # Configure logger
    logging.basicConfig(level=logging.INFO,  # Set logging level
                        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
                        datefmt='%Y-%m-%d %H:%M:%S',  # Date format without milliseconds
                        handlers=[logging.FileHandler('log.log', mode='w'),  # Log file output
                                  logging.StreamHandler()])  # Console output

    logger = logging.getLogger(__name__)

    # initializing stopping criteria
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)

    print(f'cwd: {os.getcwd()}')
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} Length of data loader: {len(train_data_loader)}")
    logger.info(f"Length of data loader: {len(train_data_loader)}")
    # training loop
    for epoch in range(opt.niter):
        model.train()
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                      "Train loss: {} at step: {} lr {}".format(model.loss, model.total_steps, model.lr))
                logger.info(f"Train loss: {model.loss} at step: {model.total_steps} lr {model.lr}")
                train_writer.add_scalar('loss', model.loss, model.total_steps)


        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            logger.info(f'saving the model at the end of epoch {epoch}')
            model.save_networks('latest')
            model.save_networks(epoch)

        # Validation
        model.eval()
        acc, ap, r_acc, f_acc = validate(model.model, val_opt)
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} (Val @ epoch {epoch}) acc: {acc}; ap: {ap} r_acc: {r_acc}; f_acc: {f_acc}")
        logger.info(f"(Val @ epoch {epoch}) acc: {acc}; ap: {ap} r_acc: {r_acc}; f_acc: {f_acc}")
        model.train()
        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                logger.info(f"changing lr at the end of epoch {epoch}, iters {model.total_steps}")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break

    model.save_networks('last')