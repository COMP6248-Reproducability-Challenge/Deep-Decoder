import torch 
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image
import PIL
import numpy as np

import matplotlib.pyplot as plt

def np_tensor(image_numpy):
    #convert image to torch.Tensor
    image_tensor = torch.from_numpy(image_numpy).unsqueeze(0)
    return image_tensor

def tensor_numpy(image_tensor):
    #convert image to np.array
    image_numpy = image_tensor.data.cpu().numpy()[0]
    return image_numpy

def pil_numpy(image_pil):
    #convert pil image to np.array
    image_numpy = np.array(image_pil)
    image_length =len(image_numpy.shape)
    # transpose image
    if image_length == 3:
        image_numpy = image_numpy.transpose(2,0,1)
        image_numpy = image_numpy.astype(np.float32) / 255
    else:
        image_numpy = image_numpy[None,...]
        image_numpy = image_numpy.astype(np.float32) / 255
        
    return image_numpy

def parameters_number(net):
    num = sum([x.numel() for x in net.parameters()])
    return num
    
def psnr(img_estimation, img_true, maximum=1.):
    img_estimation = img_estimation.flatten()
    img_true = img_true.flatten()
    error = img_estimation - img_true
    psnr = 10.*np.log(maximum**2/(np.mean(np.square(error))))/np.log(10.)
    return psnr

def comparison_show(img_origin, img_dd, img_wvl):
    psnr_dd = psnr(img_origin, img_dd)
    psnr_wvl = psnr(img_origin, img_wvl)
    
    fig = plt.figure(figsize = (15,15)) 
    ax = fig.subplots(1,3)

    if img_origin.shape[0] == 1:
        ax[0].imshow(np.clip(img_origin[0], 0, 1))
    else:
        ax[0].imshow(np.clip(img_origin.transpose(1, 2, 0), 0, 1))
    ax[0].set_title('Original image')
    ax[0].axis('off')
    
    if img_origin.shape[0] == 1:
        ax[1].imshow(np.clip(img_dd[0], 0, 1))
    else:
        ax[1].imshow(np.clip(img_dd.transpose(1, 2, 0), 0, 1))
    ax[1].set_title("Deep Decoder output image (PSNR: %.2fdB)" % psnr_dd)
    ax[1].axis('off')

    if img_origin.shape[0] == 1:
        ax[2].imshow(np.clip(img_wvl[0], 0, 1))
    else:
        ax[2].imshow(np.clip(img_wvl.transpose(1, 2, 0), 0, 1))
    ax[2].set_title("Wavelet output image (PSNR: %.2fdB)" % psnr_wvl) 
    ax[2].axis('off')
    
    plt.axis('off')
    fig.show()
