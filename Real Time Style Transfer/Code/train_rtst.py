#!/usr/bin/env python
# coding: utf-8

# In[28]:


#%matplotlib notebook

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


# In[29]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(device)


# ## Implement dataset module for COCO dataset

# In[30]:


class COCODataset(td.Dataset):

    def __init__(self, root_dir, mode='train', image_size=(256, 256)):
        super(COCODataset, self).__init__()
        self.mode = mode
        self.image_size = image_size
        self.images_dir = os.path.join(root_dir, mode + ('2015' if mode == 'test' else '2014'))
        self.files = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.files)
        #return 10

    def __repr__(self):
        return "COCODataset(mode={}, image_size={})".             format(self.mode, self.image_size)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.files[idx])
        img = Image.open(img_path).convert('RGB')
        transform = tv.transforms.Compose([
            tv.transforms.Resize(self.image_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
            ])
        img = transform(img)
        return img


# In[31]:


def myimshow(image, ax=plt):
    image = image.to('cpu').detach().numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')
    return h


# In[32]:


dataset_root_dir = '/datasets/COCO-2015'
train_set = COCODataset(dataset_root_dir, 'train')
val_set = COCODataset(dataset_root_dir, 'val')
test_set = COCODataset(dataset_root_dir, 'test')
x = test_set[8]
#myimshow(x)


# ## Define loss network

# In[33]:


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


# ##  Define image transformation network

# In[34]:


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


# In[35]:


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


# ## Train image transformation network

# In[36]:


#%matplotlib inline
style = Image.open('/datasets/ee285f-public/wikiart/wikiart/Post_Impressionism/vincent-van-gogh_irises-1889.jpg').convert('RGB')
transform = tv.transforms.Compose([
            tv.transforms.Resize((256, 256)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
            ])
style = transform(style)
#myimshow(style)


# In[37]:


content = Image.open('/datasets/ee285f-public/flickr_landscape/forest/11032118_8e196a11ea.jpg').convert('RGB')
content = transform(content)
#myimshow(content)


# In[24]:


vgg = Vgg16(requires_grad=False).to(device)
batch_size = 4
style = style.repeat(batch_size, 1, 1, 1).to(device)
features_style = vgg(normalize_batch(style))
gram_style = [gram_matrix(y) for y in features_style]


# In[ ]:





# In[25]:


net = TransformerNet(10e5, 10e11, vgg, gram_style)
net = net.to(device)
lr = 1e-3
adam = torch.optim.Adam(net.parameters(), lr=lr)
stats_manager = nt.StatsManager()
exp = nt.Experiment(net, train_set, val_set, adam, stats_manager,
                output_dir="RTST", batch_size = batch_size, perform_validation_during_training = False)


# In[26]:


def plot(exp, fig, axes, content):
    with torch.no_grad():
        transfer = exp.net(content[None].to(net.device))[0]
        axes[0].clear()
        axes[1].clear()
        myimshow(transfer, ax=axes[0])
        axes[0].set_title('transferred image')
        axes[1].plot([exp.history[k] for k in range(exp.epoch)], label="training loss")
        axes[1].legend()
        plt.tight_layout()
        fig.canvas.draw()


# In[27]:


fig, axes = plt.subplots(ncols=2, figsize=(7, 7))
exp.run(num_epochs=2, plot=lambda exp: plot(exp, fig=fig, axes=axes, content=content))


# In[ ]:


torch.save(exp.net, 'transformer_irises2')


# In[ ]:


#transformer = torch.load('transformer_picasso')


# In[ ]:


#myimshow(transformer(content[None].to(net.device))[0])

