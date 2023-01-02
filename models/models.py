import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()
    def mu(self, x):
        return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])
    def sigma(self, x):
        return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))
    def forward(self, x, y):
        return (self.sigma(y)*((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([2,3,0,1])
class DoubleConv_nBN(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.ada = AdaIN() 
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x,M):
        x = self.conv1(x)
        x = self.ada(x,M)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.ada(x,M)
        x = self.activation(x)
        return x
    
class Down_ada(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        
        self.conv = DoubleConv_nBN(in_channels, out_channels)


    def forward(self, x,M):
        x = self.maxpool(x)
        x =self.conv(x,M)
        
        return x

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_nBN(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_nBN(in_channels, out_channels)

    def forward(self, x1, x2,M):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x,M)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet1(nn.Module):
    def __init__(self, n_channels = 1, bilinear=False):
        super(UNet1, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv_nBN(n_channels, 32)
        self.down1 = Down_ada(32, 64)
        self.down2 = Down_ada(64, 128)
        self.down3 = Down_ada(128, 256)
        self.down4 = Down_ada(256, 512)
        factor = 2 if bilinear else 1
        self.Down_nBN = Down_ada(512, 1024 // factor)
        
        self.up0 = Up(1024, 512 // factor, bilinear)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32,1)
        
        
        self.up0_ = Up(1024, 512 // factor, bilinear)
        self.up1_= Up(512, 256 // factor, bilinear)
        self.up2_ = Up(256, 128 // factor, bilinear)
        self.up3_ = Up(128, 64 // factor, bilinear)
        self.up4_ = Up(64, 32, bilinear)
        self.outc_ = OutConv(32,4)
        
        
        self.ada = AdaIN() 
        
    def forward(self, x,M):
        
        #x = self.ada(x,M)
        
        x1 = self.inc(x,M)
        x2 = self.down1(x1,M)
        x3 = self.down2(x2,M)
        x4 = self.down3(x3,M)
        x5 = self.down4(x4,M)
        x6 = self.Down_nBN(x5,M)
        
        x6 = self.ada(x6,M)
        z1 = self.up0(x6, x5,M)
        z2 = self.up1(z1, x4,M)
        z3 = self.up2(z2, x3,M)
        z4 = self.up3(z3, x2,M)
        z5 = self.up4(z4, x1,M)
        logits1 = self.outc(z5)
        
        # print(z1.shape)
        # print(z2.shape)
        # print(z3.shape)
        # print(z4.shape)
        # print(z5.shape)
        # print('z shapes')
        
        #y = self.up0_(x6, x5)
        #print(y.shape)
        
#        z1 = self.ada(z1,M)
#        z2 = self.ada(z2,M)
#        z3 = self.ada(z3,M)
#        z4 = self.ada(z4,M)
#        z5 = self.ada(z5,M)
        
        y = self.up1_(z1, z2,M)
        y = self.up2_(y, z3,M)
        y = self.up3_(y, z4,M)
        y = self.up4_(y, z5,M)
        logits2 = self.outc_(y)

        return logits1,logits2

# Input_Image_Channels = 1
# def model() -> UNet:
#     model = UNet()
#     return model
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 256,256),(Input_Image_Channels,1,2)])



### the second model UNet2 Generate 1024x8x8 one-hot encode and adaptive-ins-norm ####

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        
        return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))

    def forward(self, x, y):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        return (self.sigma(y)*((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([2,3,0,1])
    
class DoubleConv_nBN(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.ada = AdaIN() 
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x,M):
        x = self.conv1(x)
        x = self.ada(x,M)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.ada(x,M)
        x = self.activation(x)
        return x
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            #nn.InstanceNorm2d(mid_channels,affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            #nn.InstanceNorm2d(out_channels,affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down_ada(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        
        self.conv = DoubleConv_nBN(in_channels, out_channels)


    def forward(self, x,M):
        x = self.maxpool(x)
        x =self.conv(x,M)
        
        return x

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet2(nn.Module):                                        ##### Generate 1024x8x8 one-hot encode and adaptive-ins-norm
    def __init__(self, n_channels = 1, bilinear=False):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down5 = Down(512, 1024 // factor)
        
        self.up0 = Up(1024, 512 // factor, bilinear)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32,1)
        
        
        self.up0_ = Up(1024, 512 // factor, bilinear)
        self.up1_= Up(512, 256 // factor, bilinear)
        self.up2_ = Up(256, 128 // factor, bilinear)
        self.up3_ = Up(128, 64 // factor, bilinear)
        self.up4_ = Up(64, 32, bilinear)
        self.outc_ = OutConv(32,4)
        
        
        self.ada = AdaIN() 
        
    def forward(self, x,M):
        
        #x = self.ada(x,M)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        
        x6 = self.ada(x6,M)

        z1 = self.up0(x6, x5)
        z2 = self.up1(z1, x4)
        z3 = self.up2(z2, x3)
        z4 = self.up3(z3, x2)
        z5 = self.up4(z4, x1)
        logits1 = self.outc(z5)
        
        # print(z1.shape)
        # print(z2.shape)
        # print(z3.shape)
        # print(z4.shape)
        # print(z5.shape)
        # print('z shapes')
        
        #y = self.up0_(x6, x5)
        #print(y.shape)
        
#        z1 = self.ada(z1,M)
#        z2 = self.ada(z2,M)
#        z3 = self.ada(z3,M)
#        z4 = self.ada(z4,M)
#        z5 = self.ada(z5,M)
        
        y = self.up1_(z1, z2)
        y = self.up2_(y, z3)
        y = self.up3_(y, z4)
        y = self.up4_(y, z5)
        logits2 = self.outc_(y)

        return logits1,logits2
        
#### The third Unet3, Generate 1024x8x8 one-hot encode and convolution before adaptive-ins-norm 
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        
        return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))

    def forward(self, x, y):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        return (self.sigma(y)*((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([2,3,0,1])
    
class DoubleConv_nBN(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.ada = AdaIN() 
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x,M):
        x = self.conv1(x)
        x = self.ada(x,M)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.ada(x,M)
        x = self.activation(x)
        return x
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            #nn.InstanceNorm2d(mid_channels,affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            #nn.InstanceNorm2d(out_channels,affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down_ada(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        
        self.conv = DoubleConv_nBN(in_channels, out_channels)


    def forward(self, x,M):
        x = self.maxpool(x)
        x =self.conv(x,M)
        
        return x

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class M_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(M_conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet3(nn.Module):                                   ##### Generate 1024x8x8 one-hot encode and convolve before adaptive-ins-norm
    def __init__(self, n_channels = 1, bilinear=False):
        super(UNet3, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down5 = Down(512, 1024 // factor)
        
        self.up0 = Up(1024, 512 // factor, bilinear)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32,1)
        
        
        self.up0_ = Up(1024, 512 // factor, bilinear)
        self.up1_= Up(512, 256 // factor, bilinear)
        self.up2_ = Up(256, 128 // factor, bilinear)
        self.up3_ = Up(128, 64 // factor, bilinear)
        self.up4_ = Up(64, 32, bilinear)
        self.outc_ = OutConv(32,4)
        
        
        self.ada = AdaIN() 
        self.M_CONV = M_conv(1024,1024)
        
    def forward(self, x,M):
        
        #x = self.ada(x,M)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        
        m_conv = self.M_CONV(M) 
        
        # print(m_conv.shape)
        
        x6 = self.ada(x6,m_conv)

        z1 = self.up0(x6, x5)
        z2 = self.up1(z1, x4)
        z3 = self.up2(z2, x3)
        z4 = self.up3(z3, x2)
        z5 = self.up4(z4, x1)
        logits1 = self.outc(z5)
        
        # print(z1.shape)
        # print(z2.shape)
        # print(z3.shape)
        # print(z4.shape)
        # print(z5.shape)
        # print('z shapes')
        
        #y = self.up0_(x6, x5)
        #print(y.shape)
        
#        z1 = self.ada(z1,M)
#        z2 = self.ada(z2,M)
#        z3 = self.ada(z3,M)
#        z4 = self.ada(z4,M)
#        z5 = self.ada(z5,M)
        
        y = self.up1_(z1, z2)
        y = self.up2_(y, z3)
        y = self.up3_(y, z4)
        y = self.up4_(y, z5)
        logits2 = self.outc_(y)

        return logits1,logits2
    
# Input_Image_Channels = 1
# def model() -> UNet3:
#     model = UNet3()
#     return model
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 256,256),(1024,8,8)])
