#!/usr/bin/python

# Residual Block


# In[ ]:


import torch
import torch.nn as nn
import math
import numpy as np

class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=192, kernel_size=1, stride=1, bias=False)
        self.in1 = nn.InstanceNorm2d(192, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=192, out_channels=154, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=154, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(32, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv3(self.conv2(output)))
        output = torch.add(output,identity_data)
        return output 


# In[ ]:


class _srmodel(nn.Module):
    def __init__(self):
        super(_srmodel, self).__init__()
        self.conv_c1to3 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.residual = self.make_layer(_Residual_Block, 1)
        self.conv_shallow = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_deep = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)

        # add the two layer

        self.upscale4x = nn.PixelShuffle(4)

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(1.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)


    def forward(self, x):

        out = self.conv_c1to3(x)
        out = self.conv_input(out)
        residual = out
        out = self.residual(out)
        out = self.conv_deep(out)
        out = self.upscale4x(out)
        residual = self.conv_shallow(residual)
        residual = self.upscale4x(residual)
        out = torch.add(out,residual)


        return out

