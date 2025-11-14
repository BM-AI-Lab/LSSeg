import torch
from torch import nn

from models.layers import RFCADenseBlock, UpSample, TransBlock, LKP
from models.ska import SKA
from utils import Smish


class FoL(nn.Module):
    """
    Focus Locally (FoL) block
    """
    def __init__(self, in_channels):
        super(FoL, self).__init__()
        assert in_channels % 8 == 0, 'in_channels must be a multiple of eight.'
        self.lkp = LKP(in_channels, lks=7, sks=3, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x


class GSC(nn.Module):
    """
    the Gated Skip Connection
    """
    def __init__(self, in_channels):
        super(GSC, self).__init__()
        self.gate_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.smish = Smish()

    def forward(self, X, Y):
        merged = torch.cat([X, Y], dim=1)
        gate = self.sigmoid(self.gate_conv(merged))
        Xgsc = self.smish(X * gate + Y * (1 - gate))
        return Xgsc


class FDM(nn.Module):
    """
    the Feature Decimation Module
    """
    def __init__(self, num_convs, in_channels, out_channels):
        super(FDM, self).__init__()
        self.dense_block = RFCADenseBlock(num_convs, in_channels, num_channels=out_channels)
        self.trans_block = TransBlock(num_convs * out_channels + in_channels, out_channels)
    
    def forward(self, X):
        return self.trans_block(self.dense_block(X))


class RM(nn.Module):
    """
    the Reconstruction Module
    """
    def __init__(self, in_channels, out_channels, upscale=2):
        super(RM, self).__init__()
        self.fol = FoL(in_channels)
        self.dpuu1 = UpSample(in_channels, out_channels, upscale)
        self.dpuu2 = UpSample(in_channels, out_channels, upscale)
        self.fuse = nn.Conv2d(2 * out_channels, out_channels, kernel_size=5, padding=2)
    
    def forward(self, X):
        X1 = self.dpuu1(self.fol(X))    # path 1
        X2 = self.dpuu2(X)              # path 2
        return self.fuse(torch.cat([X1, X2], dim=1))



class LSSeg(nn.Module):
    """
    the architecture of Line-like Structures Segmentation Network
    Params:
        in_channels: the no. channels of input images.
        len(in_channels): the no. FDM and RM pairs.
    """
    def __init__(self, in_channels):
        super(LSSeg, self).__init__()
        self.K = len(in_channels)
        self.FDMs = nn.ModuleList()
        self.RMs = nn.ModuleList()
        self.GSCs = nn.ModuleList()
        
        for i in range(self.K):
            # build FDMs
            if i != self.K - 1:
                self.FDMs.append(FDM(4, in_channels=in_channels[i], out_channels=in_channels[i + 1]))
            else:
                self.FDMs.append(FDM(4, in_channels=in_channels[i], out_channels=in_channels[i]))
            
            # build RMs
            if i == 0:
                self.RMs.append(RM(in_channels=in_channels[i + 1], out_channels=1))
            elif i == self.K - 1:
                self.RMs.append(RM(in_channels=in_channels[i], out_channels=in_channels[i]))
            else:
                self.RMs.append(RM(in_channels=in_channels[i + 1], out_channels=in_channels[i]))
            
            # build GSCs
            if i != 0:
                self.GSCs.append(GSC(in_channels[i]))

        self.af = Smish()
        self.apply(self.init_weights)

    def forward(self, X):
        X_Fs = []
        for i in range(self.K):
            X = self.FDMs[i](X)
            X_Fs.append(X)

        for i in range(self.K - 1, -1, -1):
            if i == self.K - 1:
                X = self.RMs[i](X_Fs[i])
            else:
                X = self.RMs[i](self.GSCs[i](X_Fs[i], X))
                 
        return self.af(X)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_in')
