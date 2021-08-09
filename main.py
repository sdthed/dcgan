import torch
import torch.nn as nn
from dataclasses import dataclass

from utils import get_dataloader, preview, init_weights, Config
from model import Generator, Discriminator
from train import train

# DATA_DIR = "data/celeb"


def main():
    c = Config()

    ### Data
    dataloader = get_dataloader(c.DATA_DIR, c.BATCH_SIZE, c.IM_SIZE)
    # real_batch = next(iter(dataloader))
    # preview(real_batch[0])
    # return

    ### Init Gernerator and Discriminator
    netG = Generator(c.N_GPU, c.N_Z, c.N_CHAN, c.N_GF).to(c.DEVICE)
    netG.apply(init_weights)
    netD = Discriminator(c.N_GPU, c.N_CHAN, c.N_DF)
    netD.apply(init_weights)

    ### Loss and optimizer
    criterion = nn.BCELoss()

    lr = c.LEARNING_RATE
    beta1 = c.BETA_1
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    ### Training
    train(dataloader, netG, netD, optimizerG, optimizerD, criterion, c)
    print("don't you ever get stuck in the sky")


if __name__ == "__main__":
    main()
