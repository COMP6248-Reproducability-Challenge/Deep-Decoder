import numpy as np
import numbers
import pywt
import scipy
import skimage.color as color
from skimage import data, img_as_float

def wavelet_recover(image, num_coeffs = None, mode='soft', wavelet_levels=None):
    # #创建小波对象（db1）
    wavelet = pywt.Wavelet('db1')

    a=image.shape[0]
    b=image.shape[1]

    # 确定分解的level
    if wavelet_levels is None:
        dlen = wavelet.dec_len
        l1=pywt.dwt_max_level(a, dlen)
        l2=pywt.dwt_max_level(b, dlen)
        wavelet_levels = np.min([l1,l2])
        wavelet_levels = max(wavelet_levels - 3, 1)

    all_coeffs = pywt.wavedecn(image, wavelet=wavelet, level=wavelet_levels)
    # #返回值：[cA_n, cD_n, cD_n-1, …, cD_2, cD_1]
    # #列表形式，n与分解level相关(例如level=5时，有6个返回值)，
    # #CA_n是平均系数（approximation coefficients），
    # #CD_n~CD_1是细节系数（details coefficients）

    detail_coeffs = all_coeffs[1:]
  
    alldata = []
    for level in detail_coeffs:
        for key in level:
            level[key]=level[key].tolist()
            alldata=np.append(alldata,level[key])
            
    alldata = np.abs(alldata)
    alldata = np.sort(alldata)    
    sh = all_coeffs[0].shape
    basecoeffs = sh[0]*sh[1]
    threshold = alldata[- (num_coeffs - basecoeffs)]
    
    # #根据threshold重新计算系数（根据不同的模型）
    denoised_detail = [{key: pywt.threshold(level[key],value=threshold,
                                mode=mode) for key in level} for level in detail_coeffs]
   
    denoised_coeffs = [all_coeffs[0]] + denoised_detail
    parameter=np.array(pywt.waverecn(denoised_coeffs, wavelet))[:a,:b]
    return parameter

def denoise_wavelet(image, num_coeffs=None, mode='hard', wavelet_levels=None,
                    multichannel=False, to_ycbcr=False):
    image = img_as_float(image)
    
    if multichannel:
        if to_ycbcr:
            out = color.rgb2ycbcr(image.transpose(1,2,0))
            for i in range(3):
                min, max = out[..., i].min(), out[..., i].max()
                channel = out[..., i] - min
                channel /= max - min
                out[..., i] = denoise_wavelet(channel,num_coeffs=num_coeffs,
                                              mode=mode,
                                              wavelet_levels=wavelet_levels)
                out[..., i] = out[..., i] * (max - min)
                out[..., i] += min
            out = color.ycbcr2rgb(out)
            out = out.transpose(2,0,1)
        else:
            out = np.empty_like(image)
            for c in range(image.shape[-1]):
                out[..., c] = wavelet_recover(image[..., c],num_coeffs=num_coeffs,
                                                  mode=mode,
                                                 wavelet_levels=wavelet_levels)
    else:
        out = wavelet_recover(image, mode=mode,num_coeffs=num_coeffs,
                                 wavelet_levels=wavelet_levels)

    if (image.min() < 0):        
        reconstruct_image_np = np.clip(out, *(-1,1))
    else:
        reconstruct_image_np = np.clip(out, *(0,1))
    return reconstruct_image_np


