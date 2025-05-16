import os
import sys
import time
import torch
from tensorboardX import SummaryWriter
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from util import Logger, EarlyStopping
import random
import logging


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    seed_torch(100)
    opt.dataroot = f'{opt.dataroot}/{opt.train_split}/'
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'logging.log'))
    print('  '.join(list(sys.argv)))
    val_opt = get_val_opt()
    val_opt.dataroot = f'{val_opt.dataroot}/{val_opt.val_split}/'

    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    model = Trainer(opt)
    model.train()

    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)

    # Configure logger
    logging.basicConfig(level=logging.INFO,  # Set logging level
                        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
                        datefmt='%Y-%m-%d %H:%M:%S',  # Date format without milliseconds
                        handlers=[logging.FileHandler('log.log', mode='w'),  # Log file output
                                  logging.StreamHandler()])  # Console output

    logger = logging.getLogger(__name__)

    print(f'cwd: {os.getcwd()}')
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} Length of data loader: {len(data_loader)}")
    logger.info(f"Length of data loader: {len(data_loader)}")

    for epoch in range(opt.niter):
        for i, data in enumerate(data_loader):
            model.total_steps += 1

            # Move data to cuda
            model.set_input(data)
            # Train
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} Train loss: {model.loss} at step: {model.total_steps} lr {model.lr}")
                logger.info(f"Train loss: {model.loss} at step: {model.total_steps} lr {model.lr}")
                train_writer.add_scalar('loss', model.loss, model.total_steps)

        if epoch % opt.save_epoch_freq == 0 and epoch != 0:
            # Optionally save model at each epoch
            print(f'saving the model at the end of epoch {epoch}')
            logger.info(f'saving the model at the end of epoch {epoch}')
            model.save_networks(epoch)

        if epoch % 10 == 0 and epoch != 0:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} changing lr at the end of epoch {epoch}, iters {model.total_steps}")
            logger.info(f"changing lr at the end of epoch {epoch}, iters {model.total_steps}")
            model.adjust_learning_rate()

        # Validation
        model.eval()
        acc, ap, r_acc, f_acc = validate(model.model, val_loader)
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} (Val @ epoch {epoch}) acc: {acc}; ap: {ap} r_acc: {r_acc}; f_acc: {f_acc}")
        logger.info(f"(Val @ epoch {epoch}) acc: {acc}; ap: {ap} r_acc: {r_acc}; f_acc: {f_acc}")

        model.train()

    model.eval()
    model.save_networks('last')

