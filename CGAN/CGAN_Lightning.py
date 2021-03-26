from collections import OrderedDict

import torch, utils
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from pytorch_lightning.core import LightningModule

import os
import wandb

class generator(nn.Module):

    def __init__(self, input_dim=100, output_dim=1, input_size=28, class_num=10):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, z, label):
        x = torch.cat([z, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):

    def __init__(self, input_dim=1, output_dim=1, input_size=28, class_num=10):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim + self.class_num, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, z, label):
        x = torch.cat([z, label], 1)
        x = self.conv(x)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x


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

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        data_loader = DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transform),
                                 batch_size=self.batch_size, shuffle=True)
        return data_loader

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