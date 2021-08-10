import torch
import torch.nn as nn
from dataclasses import dataclass

from utils import get_dataloader, preview, init_weights, Config
from model import Generator, Discriminator
from train import train

# DATA_DIR = "data/celeb"


def main():
    c = Config()
    c.IM_SIZE = 128
    c.BATCH_SIZE = 16

    ### Data
    dataloader = get_dataloader(c.DATA_DIR, c.BATCH_SIZE, c.IM_SIZE)
    # real_batch = next(iter(dataloader))
    # preview(real_batch[0])
    # return

    ### Init Gernerator and Discriminator
    print("init GEN")
    print("=" * 20)
    netG = Generator(c.N_Z, c.N_CHAN, c.N_GF, c.IM_SIZE).to(c.DEVICE)
    netG.apply(init_weights)
    print("init DIS")
    print("=" * 20)
    netD = Discriminator(c.N_CHAN, c.N_DF, c.IM_SIZE).to(c.DEVICE)
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


def predict():
    c = Config()
    netG = Generator(c.N_GPU, c.N_Z, c.N_CHAN, c.N_GF).to(c.DEVICE)
    # netG.apply(init_weights)
    netG.load_state_dict(torch.load("data/weights/gen_0.pth"))

    noise = torch.randn(c.BATCH_SIZE, c.N_Z, 1, 1, device=c.DEVICE)
    print(noise.shape)
    fake = netG(noise).detach().cpu()
    print(fake.shape)
    preview(fake)


def test_gen():
    c = Config()
    c.BATCH_SIZE = 4
    noise = torch.randn(c.BATCH_SIZE, c.N_Z, 1, 1, device=c.DEVICE)
    gen = Generator(c.N_Z, c.N_CHAN, c.N_GF).to(c.DEVICE)
    gen.apply(init_weights)
    out = gen(noise)
    print(out.shape)


def test_dis():
    c = Config()
    im_size = 512
    batch = torch.randn((4, 3, im_size, im_size))
    dis = Discriminator(c.N_CHAN, c.N_DF, im_size).to(c.DEVICE)
    dis.apply(init_weights)
    out = dis(batch)
    print(out.shape)


if __name__ == "__main__":
    # test_dis()
    # test_gen()
    # predict()
    main()
