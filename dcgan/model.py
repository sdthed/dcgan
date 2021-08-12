import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self, f_in, f_out, filters=4, strides=2, padding=1, has_batch_norm=True
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(f_in, f_out, filters, strides, padding, bias=False)
        self.relu = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm2d(f_out) if has_batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x) if self.bn is not None else x
        return x


class TransposeBlock(nn.Module):
    def __init__(self, f_in, f_out, filters=4, strides=2, padding=1):
        super(TransposeBlock, self).__init__()
        self.conv_t = nn.ConvTranspose2d(f_in, f_out, filters, strides, padding)
        self.bn = nn.BatchNorm2d(f_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_t(x)
        x = self.bn(x)
        return self.relu(x)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf, im_size=32):
        super(Discriminator, self).__init__()
        self.im_size = im_size
        # Block0 - Block2
        # in shape (bs, nc, width, height) ->  out shape (bs, ndf*8, widht/32, height/32)
        self.Block0 = ConvBlock(nc, ndf, has_batch_norm=False)
        self.Block1 = ConvBlock(ndf, ndf * 2)
        self.Block2 = ConvBlock(ndf * 2, ndf * 8)

        # Block3 - Block6 (Optional)
        # in shape (bs, ndf*8, widht/32, height/32)
        # out shape (bs, ndf*8, widht/512, height/512)
        self.Block3 = ConvBlock(ndf * 8, ndf * 8)
        self.Block4 = ConvBlock(ndf * 8, ndf * 8)
        self.Block5 = ConvBlock(ndf * 8, ndf * 8)
        self.Block6 = ConvBlock(ndf * 8, ndf * 8)

        # Final (run final conv layer to reduce output to vector)
        # in shape (bs, ndf*8, 4, 4) ->  out shape (bs, 1, 1, 1)
        self.conv_final = nn.Conv2d(ndf * 8, 1, 4, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.Block0(x)
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x) if self.im_size > 32 else x
        x = self.Block4(x) if self.im_size > 64 else x
        x = self.Block5(x) if self.im_size > 128 else x
        x = self.Block6(x) if self.im_size > 256 else x

        x = self.conv_final(x)
        x = self.sigmoid(x)
        return x


class Generator(nn.Module):
    def __init__(self, nz, nc, ngf, im_size=32):
        super(Generator, self).__init__()
        self.im_size = im_size

        # Block0.
        # in shape (bs, nz, 1, 1) -> out shape (bs, ngf*8, 4, 4)
        self.Block0 = TransposeBlock(nz, ngf * 8, 4, 1, 0)

        # Block1 - Block3
        # in shape (bs, ngf*8, 4, 4) -> out shape (bs, ngf, 32, 32)
        self.Block1 = TransposeBlock(ngf * 8, ngf * 4)
        self.Block2 = TransposeBlock(ngf * 4, ngf * 2)
        self.Block3 = TransposeBlock(ngf * 2, ngf)

        # Block4 - Block7 (Optional)
        # in shape (bs, ngf, 32, 32) -> out shape (bs, ngf, widht, height)
        self.Block4 = TransposeBlock(ngf, ngf)
        self.Block5 = TransposeBlock(ngf, ngf)
        self.Block6 = TransposeBlock(ngf, ngf)
        self.Block7 = TransposeBlock(ngf, ngf)

        # Final (run conv layer to reduce number of channels)
        # in shape(bs, ngf, width, height) -> out shape (bs, nc, width, height)
        self.conv_final = nn.Conv2d(ngf, nc, 1, 1, 0, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.Block0(x)
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x = self.Block4(x) if self.im_size > 32 else x
        x = self.Block5(x) if self.im_size > 64 else x
        x = self.Block6(x) if self.im_size > 128 else x
        x = self.Block7(x) if self.im_size > 256 else x

        x = self.conv_final(x)
        x = self.tanh(x)
        return x
