import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F


#Attention-based Feature Aggregation Module
class Attention(nn.Module):
    def __init__(self):
        super(Attention,self).__init__()

        self.conv1=nn.Conv2d(6, 3 , 1,bias=True)
        self.conv2=nn.Conv2d(3 , 1 ,3 , 1 ,1,bias=True)
        self.Th = nn.Sigmoid()

    def forward(self, x, y):

        res = torch.cat([x, y], dim=1)
        x1 = self.conv1(res)
        x2 = self.conv2(x1)
        x2 = self.Th(x2)
        out= x2 *  x1

        return out


class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1,bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self,x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        out=self.Conv_BN_ReLU_2(x)
        out_2=self.downsample(out)
        return out,out_2

class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(inplace=True)
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1,bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self,x,out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        x_out=self.Conv_BN_ReLU_2(x)
        x_out=self.upsample(x_out)
        cat_out=torch.cat((x_out,out),dim=1)
        return cat_out


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        out_channels=[2**(i+6) for i in range(5)] #[64, 128, 256, 512, 1024]

        #下采样
        self.d1=DownsampleLayer(6,out_channels[0])#6-64
        self.d2=DownsampleLayer(out_channels[0],out_channels[1])#64-128
        self.d3=DownsampleLayer(out_channels[1],out_channels[2])#128-256
        self.d4=DownsampleLayer(out_channels[2],out_channels[3])#256-512

        #上采样
        self.u1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
        self.u2=UpSampleLayer(out_channels[4],out_channels[2])#1024-512-256
        self.u3=UpSampleLayer(out_channels[3],out_channels[1])#512-256-128
        self.u4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64

        #输出
        self.o=nn.Sequential(

            nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1,bias=True),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True),
            #nn.Conv2d(out_channels[0],3,3,1,1)
            nn.Conv2d(out_channels[0],3,1),
            nn.Tanh()

        )

    def forward(self,x):

        #下采样
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        out_4,out4=self.d4(out3)
        #上采样
        out5=self.u1(out4,out_4)
        out6=self.u2(out5,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
        #输出
        out=self.o(out8)

        return out



if __name__ =="__main__":
    up=UNet()
    x=torch.randn(1,6,256,256)
    res=up(x)
    print(x)
    print(up)
    print(res.size())
