import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from networks.Tiny_LaDeDa import tiny_ladeda
from networks.base_model import BaseModel, init_weights
from options.train_options import TrainOptions
import random


class TinyLaDeDaTrainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt, n_classes):
        super(TinyLaDeDaTrainer, self).__init__(opt)
        self.model = tiny_ladeda(pretrained=False, preprocess_type=opt.tiny_preprocess, num_classes=n_classes)
        self.model.fc = nn.Linear(opt.features_dim, n_classes)
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
        self.loss_fn = nn.MSELoss()
        # initialize optimizers
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=0.0002, betas=(0.9, 0.999))
        self.model.to(opt.gpu_ids[0])

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.9
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*' * 25)
        print(f'Changing lr from {param_group["lr"] / 0.9} to {param_group["lr"]}')
        print('*' * 25)
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()