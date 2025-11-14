import torch
from torch import nn

from models.rfa_conv import RFCAConv
from utils import Smish


class RFCADenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        """
        Dense block with RFCA added in the first layer
        num_convs:      Number of convolutional blocks in the dense block
        input_channels: Number of input channels
        num_channels:   Number of output channels per convolutional block
        """
        super(RFCADenseBlock, self).__init__()
        layers = [RFCAConv(input_channels, num_channels, kernel_size=3, stride=1)]
        for i in range(1, num_convs):
            layers.append(self.conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layers)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            Smish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat([X, Y], dim=1)    # BxCxHxW
        return X


class DWSConv(nn.Module):
    """
    Depthwise Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DWSConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                        stride=stride, padding=kernel_size // 2, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, X):
        return self.pointwise_conv(self.depthwise_conv(X))


class TransBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        """
        Reduce height and width by half
        input_channels:  Number of input channels to the transition layer
        output_channels: Number of output channels from the transition layer
        """
        super(TransBlock, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            Smish(),
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, X):
        return self.net(X)


class DownSample(nn.Module):
    """
    TImE downsampling block, output height and width reduced by half
    num_convs: Number of convolutional layers in the dense block
    """
    def __init__(self, num_convs, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.dense_block = RFCADenseBlock(num_convs, in_channels, num_channels=out_channels)
        self.trans_block = TransBlock(num_convs * out_channels + in_channels, out_channels)
    
    def forward(self, X):
        return self.trans_block(self.dense_block(X))


class UpSample(nn.Module):
    """
    TImE upsampling block, output height and width doubled
    num_channels: Number of output channels for DWS convolution, must be a multiple of 4
    """
    def __init__(self, in_channels, out_channels, upscale=2):
        super(UpSample, self).__init__()
        self.dws_conv = DWSConv(in_channels, out_channels * upscale**2, kernel_size=3)
        self.ps = nn.PixelShuffle(upscale_factor=upscale)
        self.af = Smish()
    
    def forward(self, X):
        return self.ps(self.dws_conv(self.af(X)))


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = Smish()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        
        self.sks = sks
        self.groups = groups
        self.dim = dim
        
    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w