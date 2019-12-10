import os
import numpy as np
import torch
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt
from architecture import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def transform(image):
    image_size=(512, 512)
    transformation = tv.transforms.Compose([
            tv.transforms.Resize(image_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
    transformed_image = transformation(image)
    return(transformed_image)

def myimshow(image, ax=plt):
    image = image.to('cpu').detach().numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')
    return h

def load_image(path, device):
    img = Image.open(path).convert('RGB')
    img = transform(img)
    img = img.to(device)
    return img

def display_style_transformation(content_path, style_path, transformer_path, size, device):
    style = load_image(style_path, device)
    content = load_image(content_path, device)
    transformer = torch.load(transformer_path)
    with torch.no_grad():
        output = transformer(content[None].to(device))[0]
    fig, axes = plt.subplots(ncols = 3, figsize = (16,16))
    myimshow(content, axes[0])
    axes[0].set_title("Content image")
    myimshow(output, axes[1])
    axes[1].set_title("Stylized output")
    myimshow(style, axes[2])
    axes[2].set_title("Style image")
    
def display_RTST_styles(img_paths, device):
    l = len(img_paths)
    fig,axes = plt.subplots(ncols=l, nrows=1, figsize=(4*l,4))
    for i in range(len(img_paths)):
        style = load_image(img_paths[i], device)
        myimshow(style,axes[i])
        
def visualize_style(content_paths,style_name,device):
    l = len(content_paths)
    fig,axes = plt.subplots(ncols=l, nrows=2, figsize=(l*4,8))
    
    if style_name == 'cezanne':
        transformer = torch.load('transformer_cezanne')
    elif style_name == 'monet':
        transformer = torch.load('transformer_monet')
    elif style_name == 'irises':
        transformer = torch.load('transformer_irises')
    else:
        transformer = torch.load('transformer_starrynight')
    for i in range(l):
        content = load_image(content_paths[i],device)
        myimshow(content,axes[0][i])
        myimshow(transformer(content[None].to(device))[0],axes[1][i])
    filename = style_name+'transformed.jpg'
    #plt.savefig(filename)    
    
def visualize_all_styles(content_paths,device):
    style_cezanne =load_image('s_cezannePost_Impressionism_paul-cezanne_forest.jpg',device)
    style_irises = load_image('s_irisesPost_Impressionism_vincent-van-gogh_irises-1889.jpg',device)
    style_starry = load_image('s_starry_nightPost_Impressionism_vincent-van-gogh_the-starry-night-1889(1).jpg',device)
    style_monet = load_image('s_monetImpressionism_claude-monet_the-bodmer-oak-fontainebleau.jpg',device)
    transformer_cezanne = torch.load('transformer_cezanne')
    transformer_irises = torch.load('transformer_irises')
    transformer_starry = torch.load('transformer_starrynight')
    transformer_monet = torch.load('transformer_monet')
    l = len(content_paths)
    fig,axes = plt.subplots(ncols=4, nrows=l, figsize=(16,l*4))
    for i in range(l):
        content = load_image(content_paths[i],device)
        myimshow(transformer_cezanne(content[None].to(device))[0],axes[i][0])
        axes[i][0].set_title("Cezanne")
        myimshow(transformer_monet(content[None].to(device))[0],axes[i][1])
        axes[i][1].set_title("Monet")
        myimshow(transformer_irises(content[None].to(device))[0],axes[i][2])
        axes[i][2].set_title("Irises")
        myimshow(transformer_starry(content[None].to(device))[0],axes[i][3])
        axes[i][3].set_title("Starry night")
    filename = 'style_comparison'+"_".join(content_paths).replace("\\",'_').replace("/",'_')+'.jpg'
    #plt.savefig(filename)
