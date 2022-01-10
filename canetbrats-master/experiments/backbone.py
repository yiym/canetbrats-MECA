from .backbone_resnet import *
from .backbone_unet_encoder import unet_encoder
import torch
import time
import torch.nn as nn
import math


class meca_block(nn.Module):

    def __init__(self, k_size=3):
        super(meca_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()



    def forward(self,x):
        b, c, h, w, d = x.size()
        avg = self.avg_pool(x)
        max = self.max_pool(x)

        avg_pool_out = self.conv(avg.squeeze(-1).squeeze(-2).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-2).unsqueeze(-1)
        max_pool_out = self.conv(max.squeeze(-1).squeeze(-2).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-2).unsqueeze(-1)

        out = avg_pool_out + max_pool_out
        out = self.sigmoid(out)
        out = out.expand_as(x)

        return out * x

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

__all__ = ['Backbone']

class Backbone(nn.Module):
    def __init__(self, backbone):
        super(Backbone, self).__init__()

        self.meca = meca_block()

        if backbone == 'resnet3d18':
            self.pretrained = resnet3d18()
        elif backbone == 'resnet3d34':
            self.pretrained = resnet3d34()
        elif backbone == 'resnet3d50':
            self.pretrained = resnet3d50()
        elif backbone == 'unet_encoder':
            self.pretrained = unet_encoder()
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        #upsample options
        self._up_kwargs = up_kwargs

    def backbone_forward(self, x):


        conv1 = self.pretrained.dconv_down1(x)
        x1 = self.meca(conv1)
        x1 = conv1 + x1
        pool1 = self.pretrained.maxpool(x1)

        conv2 = self.pretrained.dconv_down2(pool1)
        x2 = self.meca(conv2)
        x2 = conv2 + x2
        pool2 = self.pretrained.maxpool(x2)

        conv3 = self.pretrained.dconv_down3(pool2)
        x3 = self.meca(conv3)
        x3 = conv3 + x3
        pool3 = self.pretrained.maxpool(x3)

        conv4 = self.pretrained.dconv_down4(pool3)
        x4 = self.meca(conv4)
        x4 = x4 + conv4

        return x1, x2, x3, x4



