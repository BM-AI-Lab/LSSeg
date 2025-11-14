from models.lsseg import *


class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        """
        num_convs:      Number of convolutional blocks in the dense block
        input_channels: Number of input channels
        num_channels:   Number of output channels per convolutional block
        """
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_convs):
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


class FDM_no_RFCA(nn.Module):
    """
    the Feature Decimation Module
    """
    def __init__(self, num_convs, in_channels, out_channels):
        super(FDM_no_RFCA, self).__init__()
        self.dense_block = DenseBlock(num_convs, in_channels, num_channels=out_channels)
        self.trans_block = TransBlock(num_convs * out_channels + in_channels, out_channels)
    
    def forward(self, X):
        return self.trans_block(self.dense_block(X))


class RM_no_FoL(nn.Module):
    """
    the Reconstruction Module
    """
    def __init__(self, in_channels, out_channels, upscale=2):
        super(RM_no_FoL, self).__init__()
        self.dpuu1 = UpSample(in_channels, out_channels, upscale)
        self.dpuu2 = UpSample(in_channels, out_channels, upscale)
        self.fuse = nn.Conv2d(2 * out_channels, out_channels, kernel_size=5, padding=2)
    
    def forward(self, X):
        X1 = self.dpuu1(X)    # path 1
        X2 = self.dpuu2(X)    # path 2
        return self.fuse(torch.cat([X1, X2], dim=1))


class a(nn.Module):
    """
    RFCA    GSC    FoL
    no      no     no
    """
    def __init__(self, in_channels):
        super(a, self).__init__()
        self.K = len(in_channels)
        self.FDM_no_RFCAs = nn.ModuleList()
        self.RM_no_FoLs = nn.ModuleList()
        
        for i in range(self.K):
            # build FDM_no_RFCAs
            if i != self.K - 1:
                self.FDM_no_RFCAs.append(FDM_no_RFCA(4, in_channels=in_channels[i], out_channels=in_channels[i + 1]))
            else:
                self.FDM_no_RFCAs.append(FDM_no_RFCA(4, in_channels=in_channels[i], out_channels=in_channels[i]))
            
            # build RM_no_FoLs
            if i == 0:
                self.RM_no_FoLs.append(RM_no_FoL(in_channels=in_channels[i + 1], out_channels=1))
            elif i == self.K - 1:
                self.RM_no_FoLs.append(RM_no_FoL(in_channels=in_channels[i], out_channels=in_channels[i]))
            else:
                self.RM_no_FoLs.append(RM_no_FoL(in_channels=in_channels[i + 1], out_channels=in_channels[i]))

        self.af = Smish()
        self.apply(self.init_weights)

    def forward(self, X):
        for i in range(self.K):
            X = self.FDM_no_RFCAs[i](X)

        for i in range(self.K - 1, -1, -1):
            X = self.RM_no_FoLs[i](X)
        return self.af(X)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_in')


class b(nn.Module):
    """
    RFCA    GSC    FoL
    yes      no     no
    """
    def __init__(self, in_channels):
        super(b, self).__init__()
        self.K = len(in_channels)
        self.FDMs = nn.ModuleList()
        self.RM_no_FoLs = nn.ModuleList()
        
        for i in range(self.K):
            # build FDMs
            if i != self.K - 1:
                self.FDMs.append(FDM(4, in_channels=in_channels[i], out_channels=in_channels[i + 1]))
            else:
                self.FDMs.append(FDM(4, in_channels=in_channels[i], out_channels=in_channels[i]))
            
            # build RM_no_FoLs
            if i == 0:
                self.RM_no_FoLs.append(RM_no_FoL(in_channels=in_channels[i + 1], out_channels=1))
            elif i == self.K - 1:
                self.RM_no_FoLs.append(RM_no_FoL(in_channels=in_channels[i], out_channels=in_channels[i]))
            else:
                self.RM_no_FoLs.append(RM_no_FoL(in_channels=in_channels[i + 1], out_channels=in_channels[i]))

        self.af = Smish()
        self.apply(self.init_weights)

    def forward(self, X):
        for i in range(self.K):
            X = self.FDMs[i](X)

        for i in range(self.K - 1, -1, -1):
            X = self.RM_no_FoLs[i](X)
        return self.af(X)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_in')


class c(nn.Module):
    """
    RFCA    GSC    FoL
    no      yes     no
    """
    def __init__(self, in_channels):
        super(c, self).__init__()
        self.K = len(in_channels)
        self.FDM_no_RFCAs = nn.ModuleList()
        self.RM_no_FoLs = nn.ModuleList()
        self.GSCs = nn.ModuleList()
        
        for i in range(self.K):
            # build FDM_no_RFCAs
            if i != self.K - 1:
                self.FDM_no_RFCAs.append(FDM_no_RFCA(4, in_channels=in_channels[i], out_channels=in_channels[i + 1]))
            else:
                self.FDM_no_RFCAs.append(FDM_no_RFCA(4, in_channels=in_channels[i], out_channels=in_channels[i]))
            
            # build RM_no_FoLs
            if i == 0:
                self.RM_no_FoLs.append(RM_no_FoL(in_channels=in_channels[i + 1], out_channels=1))
            elif i == self.K - 1:
                self.RM_no_FoLs.append(RM_no_FoL(in_channels=in_channels[i], out_channels=in_channels[i]))
            else:
                self.RM_no_FoLs.append(RM_no_FoL(in_channels=in_channels[i + 1], out_channels=in_channels[i]))
            
            # build GSCs
            if i != 0:
                self.GSCs.append(GSC(in_channels[i]))

        self.af = Smish()
        self.apply(self.init_weights)

    def forward(self, X):
        X_Fs = []
        for i in range(self.K):
            X = self.FDM_no_RFCAs[i](X)
            X_Fs.append(X)

        for i in range(self.K - 1, -1, -1):
            if i == self.K - 1:
                X = self.RM_no_FoLs[i](X_Fs[i])
            else:
                X = self.RM_no_FoLs[i](self.GSCs[i](X_Fs[i], X))
        
        return self.af(X)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_in')


class d(nn.Module):
    """
    RFCA    GSC    FoL
    no      no     yes
    """
    def __init__(self, in_channels):
        super(d, self).__init__()
        self.K = len(in_channels)
        self.FDM_no_RFCAs = nn.ModuleList()
        self.RMs = nn.ModuleList()
        
        for i in range(self.K):
            # build FDM_no_RFCAs
            if i != self.K - 1:
                self.FDM_no_RFCAs.append(FDM_no_RFCA(4, in_channels=in_channels[i], out_channels=in_channels[i + 1]))
            else:
                self.FDM_no_RFCAs.append(FDM_no_RFCA(4, in_channels=in_channels[i], out_channels=in_channels[i]))
            
            # build RMs
            if i == 0:
                self.RMs.append(RM(in_channels=in_channels[i + 1], out_channels=1))
            elif i == self.K - 1:
                self.RMs.append(RM(in_channels=in_channels[i], out_channels=in_channels[i]))
            else:
                self.RMs.append(RM(in_channels=in_channels[i + 1], out_channels=in_channels[i]))

        self.af = Smish()
        self.apply(self.init_weights)

    def forward(self, X):
        for i in range(self.K):
            X = self.FDM_no_RFCAs[i](X)

        for i in range(self.K - 1, -1, -1):
            X = self.RMs[i](X)
        return self.af(X)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_in')


class e(nn.Module):
    """
    RFCA    GSC    FoL
    yes      yes     no
    """
    def __init__(self, in_channels):
        super(e, self).__init__()
        self.K = len(in_channels)
        self.FDMs = nn.ModuleList()
        self.RM_no_FoLs = nn.ModuleList()
        self.GSCs = nn.ModuleList()
        
        for i in range(self.K):
            # build FDMs
            if i != self.K - 1:
                self.FDMs.append(FDM(4, in_channels=in_channels[i], out_channels=in_channels[i + 1]))
            else:
                self.FDMs.append(FDM(4, in_channels=in_channels[i], out_channels=in_channels[i]))
            
            # build RM_no_FoLs
            if i == 0:
                self.RM_no_FoLs.append(RM_no_FoL(in_channels=in_channels[i + 1], out_channels=1))
            elif i == self.K - 1:
                self.RM_no_FoLs.append(RM_no_FoL(in_channels=in_channels[i], out_channels=in_channels[i]))
            else:
                self.RM_no_FoLs.append(RM_no_FoL(in_channels=in_channels[i + 1], out_channels=in_channels[i]))
            
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
                X = self.RM_no_FoLs[i](X_Fs[i])
            else:
                X = self.RM_no_FoLs[i](self.GSCs[i](X_Fs[i], X))
        
        return self.af(X)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_in')


class f(nn.Module):
    """
    RFCA    GSC    FoL
    yes      no     yes
    """
    def __init__(self, in_channels):
        super(f, self).__init__()
        self.K = len(in_channels)
        self.FDMs = nn.ModuleList()
        self.RMs = nn.ModuleList()
        
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

        self.af = Smish()
        self.apply(self.init_weights)

    def forward(self, X):
        for i in range(self.K):
            X = self.FDMs[i](X)

        for i in range(self.K - 1, -1, -1):
            X = self.RMs[i](X)
                 
        return self.af(X)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_in')


class g(nn.Module):
    """
    RFCA    GSC    FoL
    no      yes     yes
    """
    def __init__(self, in_channels):
        super(g, self).__init__()
        self.K = len(in_channels)
        self.FDM_no_RFCAs = nn.ModuleList()
        self.RMs = nn.ModuleList()
        self.GSCs = nn.ModuleList()
        
        for i in range(self.K):
            # build FDMs
            if i != self.K - 1:
                self.FDM_no_RFCAs.append(FDM_no_RFCA(4, in_channels=in_channels[i], out_channels=in_channels[i + 1]))
            else:
                self.FDM_no_RFCAs.append(FDM_no_RFCA(4, in_channels=in_channels[i], out_channels=in_channels[i]))
            
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
            X = self.FDM_no_RFCAs[i](X)
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

