from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms


def get_dataloader(data_dir, batch_size=2, im_size=64):
    dataset = dset.ImageFolder(
        root=data_dir,
        transform=transforms.Compose(
            [
                transforms.Resize(im_size),
                transforms.CenterCrop(im_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    return dataloader


def preview(batch):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Data")
    plt.imshow(
        np.transpose(
            vutils.make_grid(batch.to("cpu")[:64], padding=2, normalize=True).cpu(),
            (1, 2, 0),
        )
    )
    plt.show()


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def unzip(file_path, target_dir):
    import zipfile

    print("unzipping ...")
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    print("DONE!")


@dataclass
class Config:
    # DATA_DIR = "data"
    DATA_DIR = "data/art/images/images"

    WORKERS = 2
    BATCH_SIZE = 128
    N_EPOCHS = 5

    IM_SIZE = 64
    N_CHAN = 3
    N_Z = 100  # size of z latent vector
    N_GF = 64  # size of feature maps in generator
    N_DF = 64  # size of feature maps in discriminator

    LEARNING_RATE = 0.0002
    BETA_1 = 0.5
    N_GPU = 1

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    real_label = 1.0
    fake_label = 0.0
