import torch
import copy
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class Decoder(nn.Module):
    def __init__(self, k_channels, output_channels, upsample_times=4):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(k_channels, k_channels, (1,1), padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(k_channels, affine=True)
        self.conv2 = nn.Conv2d(k_channels, k_channels, (1,1), padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(k_channels, affine=True)
        self.conv3 = nn.Conv2d(k_channels, k_channels, (1,1), padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(k_channels, affine=True)
        self.conv4 = nn.Conv2d(k_channels, k_channels, (1,1), padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(k_channels, affine=True)
        if(upsample_times==5):
            self.conv_add = nn.Conv2d(k_channels, k_channels, (1,1), padding=0, bias=False)
            self.bn_add = nn.BatchNorm2d(k_channels, affine=True)
        else:
            self.conv_add = None
            self.bn_add = None
        self.conv5 = nn.Conv2d(k_channels, k_channels, (1,1), padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(k_channels, affine=True)
        self.conv6 = nn.Conv2d(k_channels, output_channels, (1,1), padding=0, bias=False)

        
    def forward(self, x):
        out = self.conv1(x)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.bn4(out)
        out = F.relu(out)
        if self.conv_add != None:
            out = self.conv_add(out)
            out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
            out = self.bn_add(out)
            out = F.relu(out)
        out = self.conv5(out)
        out = self.bn5(out)
        out = F.relu(out)
        out = self.conv6(out)
        out = torch.sigmoid(out)
        return out