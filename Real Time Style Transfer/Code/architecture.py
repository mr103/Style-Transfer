import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt
import nntools_RTST as nt
from collections import namedtuple
class Vgg16(torch.nn.Module):
    '''
    ref: https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/vgg.py
    '''
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = tv.models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
class TransformerNet(nt.NeuralNetwork):
    '''
    ref: https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/transformer_net.py
    '''
    def __init__(self, content_weight, style_weight, loss_network, gram_style):
        super(TransformerNet, self).__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.loss_network = loss_network
        self.gram_style = gram_style
        self.mse_loss = nn.MSELoss()
        # Initial convolution layers
        self.conv = nn.ModuleList()
        self.inp = nn.ModuleList()
        self.conv.append(ConvLayer(3, 32, kernel_size=9, stride=1))
        self.inp.append(torch.nn.InstanceNorm2d(32, affine=True))
        self.conv.append(ConvLayer(32, 64, kernel_size=3, stride=2))
        self.inp.append(torch.nn.InstanceNorm2d(64, affine=True))
        self.conv.append(ConvLayer(64, 128, kernel_size=3, stride=2))
        self.inp.append(torch.nn.InstanceNorm2d(128, affine=True))
        # Residual layers
        self.res = nn.ModuleList()
        self.res.append(ResidualBlock(128))
        self.res.append(ResidualBlock(128))
        self.res.append(ResidualBlock(128))
        self.res.append(ResidualBlock(128))
        self.res.append(ResidualBlock(128))
        # Upsampling Layers
        self.deconv = nn.ModuleList()
        self.deconv.append(UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2))
        self.inp.append(torch.nn.InstanceNorm2d(64, affine=True))
        self.deconv.append(UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2))
        self.inp.append(torch.nn.InstanceNorm2d(32, affine=True))
        self.deconv.append(ConvLayer(32, 3, kernel_size=9, stride=1))
        # Non-linearities
        self.relu = torch.nn.ReLU()
        
    def criterion(self, x, y):
        x = normalize_batch(x)
        y = normalize_batch(y)
        features_x = self.loss_network(x)
        features_y = self.loss_network(y)
        content_loss = self.content_weight * self.mse_loss(features_y.relu2_2, features_x.relu2_2)
        style_loss = 0.
        for ft_y, gm_s in zip(features_y, self.gram_style):
            gm_y = gram_matrix(ft_y)
            style_loss += self.mse_loss(gm_y, gm_s[:len(x), :, :])
        style_loss *= self.style_weight
        return content_loss + style_loss

    def forward(self, X):
        y = self.relu(self.inp[0](self.conv[0](X)))
        y = self.relu(self.inp[1](self.conv[1](y)))
        y = self.relu(self.inp[2](self.conv[2](y)))
        y = self.res[0](y)
        y = self.res[1](y)
        y = self.res[2](y)
        y = self.res[3](y)
        y = self.res[4](y)
        y = self.relu(self.inp[3](self.deconv[0](y)))
        y = self.relu(self.inp[4](self.deconv[1](y)))
        y = self.deconv[2](y)
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.ModuleList()
        self.inp = nn.ModuleList()
        self.conv.append(ConvLayer(channels, channels, kernel_size=3, stride=1))
        self.inp.append(torch.nn.InstanceNorm2d(channels, affine=True))
        self.conv.append(ConvLayer(channels, channels, kernel_size=3, stride=1))
        self.inp.append(torch.nn.InstanceNorm2d(channels, affine=True))
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.inp[0](self.conv[0](x)))
        out = self.inp[1](self.conv[1](out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out  
    
