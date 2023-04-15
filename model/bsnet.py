""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn

##### Model #####

class BSNet(nn.Module):
    def __init__(self, size, n_channels = 2, n_classes = 2, bilinear=True):
        super(BSNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        h, w = size
        #y, x = torch.meshgrid([torch.arange(h-1, -1, -1), torch.arange(0, w)])
        #x, y = x / (w - 1.), y / (h - 1.)
        #self.template = torch.stack((x, y))

        assert h == w
        assert h%8==0
        ws = int(h/8)

        self.fft = FFT(ws)
        self.dtl = DTL(ws, ws)

        self.down = Down(16, 16)
        self.inc = DoubleConv(2, 64)
        self.inc2 = DoubleConv(2, 16)
        self.up1 = Up(64, 64, bilinear)
        self.up2 = Up(64+16, 64, bilinear)
        self.up3 = Up(64, 48, bilinear)
        self.outc = OutConv(48, n_classes)

    def forward(self, x):
        x1 = self.fft(x)
        x12 = self.dtl(x1)
        x2 = self.inc2(x)
        x2 = self.down(x2)
        x = self.inc(x12)
        x = self.up1(x)
        x = self.up2(x, x2)
        x = self.up3(x)
        mapping = self.outc(x)# + self.template
        mapping[:, 0, :, 0] = 0 - 0.5
        mapping[:, 0, :, -1] = 1 - 0.5
        mapping[:, 1, 0, :] = 1 - 0.5
        mapping[:, 1, -1, :] = 0 - 0.5
        return mapping + 0.5



##### Components #####

class FFT(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.r = width//2
        
    def forward(self, x):
        N, c, rows, cols = x.shape
        assert c==2
        r = self.r
        crow, ccol = rows//2, cols//2

        x = x[:, 0] + x[:, 1] * 1j
        freq = torch.fft.fft2(x)
        freq = torch.fft.fftshift(freq)[:, crow-r:crow+r, ccol-r:ccol+r]
        freq = torch.stack((freq.real, freq.imag), dim = 1)

        return freq

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, pad=True):
        super().__init__()
        pad = 1 if pad else 0
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=pad, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, pad=True):
        super().__init__()
        pad = 1 if pad else 0
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DTL(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, padding=True):
        super().__init__()
        self.in_channels = in_channels
        assert in_channels == out_channels
        self.conv1r = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias = False)
        self.conv1i = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias = False)
        self.conv2r = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias = False)
        self.conv2i = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias = False)

    def forward(self, x):
        N = x.shape[0]
        x_pmt = x.permute((0, 2, 3, 1))
        x1r = self.conv1r(x_pmt)
        x1i = self.conv1i(x_pmt)
        x1ac = x1r[:, :, :, 0]
        x1bc = x1r[:, :, :, 1]
        x1ad = x1i[:, :, :, 0]
        x1bd = x1i[:, :, :, 1]
        real = x1ac - x1bd
        imag = x1ad + x1bc
        x2 = torch.stack((real, imag), dim=1)
        x3 = x2.permute((0, 3, 2, 1))
        x4r = self.conv2r(x3)
        x4i = self.conv2i(x3)
        x4ac = x4r[:, :, :, 0]
        x4bc = x4r[:, :, :, 1]
        x4ad = x4i[:, :, :, 0]
        x4bd = x4i[:, :, :, 1]
        real = x4ac - x4bd
        imag = x4ad + x4bc
        x5 = torch.stack((real, imag), dim=1)
        x6 = x5.permute((0, 1, 3, 2))
        return x6

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            #DoubleConv(in_channels, out_channels)
            Conv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
            #self.conv = Conv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        x = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    import numpy as np
    net = BSNet(size = [10, 10], device = 'cpu')
    x = np.arange(12800).reshape((1, 2, 80, 80))/12800
    x = torch.from_numpy(x).float()
    out = net(x)
    print(out.shape)
