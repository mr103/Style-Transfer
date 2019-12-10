from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import copy
import pandas as pd
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def demo_images():
    
    style_root_dir = '/datasets/ee285f-public/wikiart/wikiart'
    content_root_dir = '/datasets/ee285f-public/flickr_landscape'
    ##styles_1 = ['Abstract_Expressionism/elaine-de-kooning_untitled-1965.jpg',
    ##          'Impressionism/paul-cezanne_the-orchard-1877.jpg',
    ##          'Realism/winslow-homer_the-guide.jpg']
    
    styles = ['Action_painting/walasse-ting_two-angels.jpg',
              'Northern_Renaissance/rogier-van-der-weyden_the-last-judgement-1450.jpg',
              'Romanticism/william-turner_wilderness-a-engedi-convent-of-santa-saba.jpg']
    
    contents = ['forest/9628204808_5a3f1434aa.jpg',
                'city/239248438_96fceedfba.jpg']
    
    style_images = [image_loader(os.path.join(style_root_dir, x)) for x in styles]
    content_images = [image_loader(os.path.join(content_root_dir, x)) for x in contents]
    
    return style_images, content_images

def demo_style_transfer(content_images, style_images):
    
    content_name = ['Forest', 'City']
    style_name = ['Action_painting','Northern_Renaissance','Romanticism']
    total_len = len(content_images) * len(style_images)
    
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    outputs = []
    start = time.time()
    for i, content in enumerate(content_images):
        
        content = content.to(device)
        input_image = content.clone().detach().requires_grad_(True)
        for j, style in enumerate(style_images):
            
            print('{}/{}. Content: {}, Style: {} transfer start!'.format(
                (i*3+j+1), total_len, content_name[i], style_name[j]))
            print()
            
            style = style.to(device)
            output = run_style_transfer(vgg, content, style, input_image)
            outputs.append(output.clone())
            
            print('Content: {}, Style: {} transfer complete!'.format(
                content_name[i],style_name[j]))
            print('----------------------------------------------------------')
            print()
    
    time_elapsed = time.time() - start
    print('< Transfering complete in {:.0f}m {:.0f}s >'.format(
        time_elapsed // 60, time_elapsed % 60))    
    
    return outputs

img_size = 512

loader = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor()])

def image_loader(image_name):
    
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    
    return image.to(device, torch.float)

class Content_Loss(nn.Module):

    def __init__(self, target,):
        super(Content_Loss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        
        self.loss = F.mse_loss(input, self.target)
        
        return input
    
def Gram_Matrix(input):
    
    i, j, k, l = input.size()  
    features = input.view(i * j, k * l)
    G = torch.mm(features, features.t())

    # Normalize the values of the gram matrix
    return G.div(i * j * k * l)

class Style_Loss(nn.Module):

    def __init__(self, target_feature):
        super(Style_Loss, self).__init__()
        self.target = Gram_Matrix(target_feature).detach()

    def forward(self, input):
        GMtrx = Gram_Matrix(input)
        self.loss = F.mse_loss(GMtrx, self.target)
        return input

cnn_norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.as_tensor(mean).view(-1, 1, 1)
        self.std = torch.as_tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


# desired depth layers to compute style/content losses :
content_layers_default = ['conv_5']
style_layers_default = ['conv_1', 'conv_3', 'conv_5', 'conv_9', 'conv_13']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_image, content_image,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # Normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # losses
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_image).detach()
            content_loss = Content_Loss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_image).detach()
            style_loss = Style_Loss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # Trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], Content_Loss) or isinstance(model[i], Style_Loss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def run_style_transfer(cnn, content_img, style_img, input_img,
                       normalization_mean=cnn_norm_mean,
                       normalization_std=cnn_norm_std, 
                       num_steps=200, style_weight=1000000, content_weight=1):
    
    print('Building the style transfer model...')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    print('Optimizing...')
    print()
    
    start = time.time()
    
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 20 == 0:
            #if run[0] % 50 == 0:
                print("run {}/{}, Style Loss: {:4f}, Content Loss: {:4f}".format(
                    run[0], num_steps, style_score.item(), content_score.item()))

            return style_score + content_score

        optimizer.step(closure)

    # Last correction
    input_img.data.clamp_(0, 1)
    
    time_elapsed = time.time() - start
    print('Transfering complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print()

    return input_img