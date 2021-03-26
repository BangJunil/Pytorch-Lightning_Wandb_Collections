from collections import OrderedDict

import torch, utils
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.core import LightningModule

from discriminator import discriminator
from generator import generator

import os
import wandb

class CGAN(LightningModule):
    def __init__(self,
                 channels: int = 1,
                 z_dim: int = 100,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 batch_size: int = 64,
                 input_size: int = 28,
                 class_num: int = 10,
                 **kwargs):

        super().__init__()
        self.channels = channels
        self.z_dim = z_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size
        self.input_size = input_size
        self.class_num = 10
        self.sample_num = self.class_num ** 2

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=self.channels, input_size=self.input_size,
                           class_num=self.class_num)
        self.D = discriminator(input_dim=self.channels, output_dim=1, input_size=self.input_size,
                               class_num=self.class_num)
        self.G.cuda()
        self.D.cuda()

    def forward(self, z, label):
        return self.G(z, label)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_, y_ = batch
        z_ = torch.rand((x_.shape[0], self.z_dim))
        z_ = z_.type_as(x_)
        y_vec_ = torch.zeros((x_.shape[0], self.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
        y_fill_ = y_vec_.unsqueeze(2).unsqueeze(3).expand(x_.shape[0], self.class_num, self.input_size, self.input_size)
        x_, z_, y_vec_, y_fill_ = x_.cuda(), z_.cuda(), y_vec_.cuda(), y_fill_.cuda()

        # train generator
        if optimizer_idx == 0:
            self.generated_imgs = self(z_, y_vec_)
            valid = torch.ones(x_.size(0), 1)
            valid = valid.type_as(x_)

            g_loss = self.adversarial_loss(self.D(self(z_, y_vec_), y_fill_), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            valid = torch.ones(x_.size(0), 1)
            valid = valid.type_as(x_)
            real_loss = self.adversarial_loss(self.D(x_, y_fill_), valid)

            fake = torch.zeros(x_.size(0), 1)
            fake = fake.type_as(x_)
            fake_loss = self.adversarial_loss(self.D(self(z_, y_vec_), y_fill_), fake)

            d_loss = real_loss + fake_loss
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        sample_y_ = torch.zeros(self.batch_size, self.class_num).scatter_(1, torch.randint(0, self.class_num - 1,
                                                                                           (self.batch_size, 1)).type(
            torch.LongTensor), 1)
        sample_z_ = torch.rand((self.batch_size, self.z_dim))
        sample_z_, sample_y_ = sample_z_.cuda(), sample_y_.cuda()

        # log sampled images
        sample_imgs = self(sample_z_, sample_y_)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.log({'generated_images': [wandb.Image(grid.cpu(), caption=self.current_epoch)]})