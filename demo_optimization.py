#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 23:32:16 2024

@author: paviprathiraja

Colorization on a black and white image using two pre-trained deep learning models: ECCV16 and SIGGRAPH17.

Module named 'colorizers'

"""

import argparse
import matplotlib.pyplot as plt

from colorizers import *

# Load colorizers outside the loop
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

# Add argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='imgs/sea1.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

# Load image
img = load_img(opt.img_path)

# Process image
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
if opt.use_gpu:
    tens_l_rs = tens_l_rs.cuda()

# Colorize images
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

# Save colorized images
#plt.imsave('%s_eccv16.png' % opt.save_prefix, out_img_eccv16)
plt.imsave('%s_eccv16.jpg' % opt.save_prefix, out_img_eccv16)

#plt.imsave('%s_siggraph17.png' % opt.save_prefix, out_img_siggraph17)
plt.imsave('%s_siggraph17.jpg' % opt.save_prefix, out_img_siggraph17)


# Display images
plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(img_bw)
plt.title('Input')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(out_img_eccv16)
plt.title('Output (ECCV 16)')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(out_img_siggraph17)
plt.title('Output (SIGGRAPH 17)')
plt.axis('off')
plt.show()
