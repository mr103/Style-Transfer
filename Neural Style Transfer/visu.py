import matplotlib.pyplot as plt
import numpy as np

def convert(img):
    img = img.to('cpu').detach().numpy()
    img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])
    img[img < 0] = 0
    img[img > 1] = 1
    
    return img

def display(img1, img2):
    img1 = convert(img1)
    img2 = convert(img2)
    fig, ax = plt.subplots(1, 2, figsize=(9,6))
    ax[0].set_title('Style')
    ax[0].imshow(img1)
    ax[1].set_title('Content')
    ax[1].imshow(img2)
    ax[0].axis('off')
    ax[1].axis('off')
    
def display_style_content(imgs, types):
    fig, ax = plt.subplots(1, len(types), figsize=(len(types)*4,4))
    fig.suptitle('CONTENT', fontsize=14) if len(types) == 2 \
        else fig.suptitle('STYLE', fontsize=14)
    for i, style in enumerate(types):        
        img = convert(imgs[i][0])
        ax[i].set_title(style)
        ax[i].imshow(img)
        ax[i].axis('off')
        
def display_demo_result(imgs, inputs):
    
    types = ['Input','Action_painting','Northern_Renaissance','Romanticism']
    fig, ax = plt.subplots(2, 4, figsize=(16,8), constrained_layout=False)
    for i in range(2):
        ax[i][0].imshow(convert(inputs[i][0]))
        ax[i][1].imshow(convert(imgs[i*3][0]))
        ax[i][2].imshow(convert(imgs[i*3+1][0]))
        ax[i][3].imshow(convert(imgs[i*3+2][0]))
        ax[i][0].axis('off')
        ax[i][1].axis('off')
        ax[i][2].axis('off')
        ax[i][3].axis('off')
        '''
        if i == 0:
            for j in range(4):
                ax[i][j].set_title(types[j])
        '''
    plt.subplots_adjust(wspace=.001, hspace=.08)
    plt.savefig('result_for_nst.png')