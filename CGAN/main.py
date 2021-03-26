from CGAN_Lightning import CGAN
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
import os, sys
from data_loader import MNIST_dataset

def main():
    wandb.login()
    wandb.init(entity='hyeonsu')

    model = CGAN()
    dataset = MNIST_dataset()
    wandb_logger = WandbLogger(project='CGAN-Wandb')
    trainer = Trainer(gpus=1, logger=wandb_logger, max_epochs=50)

    trainer.fit(model, dataset)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    main()