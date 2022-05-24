import os
import random
import sys

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision.transforms import functional as tf

# noise and blur
kernel_size_list = [3,5,7,9]
def gau_blur(input):
    kernel_size = random.choice(kernel_size_list)
    out = tf.gaussian_blur(input, kernel_size=kernel_size)
    return out

def gau_noise(input, mean=0, noise_factor=(0.05, 0.2)):
    stdv = random.uniform(noise_factor[0], noise_factor[1])
    noise = stdv * torch.randn_like(input) + mean
    x = input + noise
    out = minmax_scale(x, max=None, min=None, new_max=1, new_min=0)
    return out

def salt_pepper(input, preserve_factor=(0.7, 0.95)):
    img_preserve_rate = random.uniform(preserve_factor[0], preserve_factor[1])
    proper_size = (1, input.shape[1], input.shape[2])

    mask = torch.rand(proper_size, device=input.device)
    noise_mask = (mask > img_preserve_rate)
    img_mask = ~noise_mask

    noise_mask = noise_mask.type(torch.float32)
    noise_mask = torch.cat([noise_mask] * 3, dim=0)

    img_mask = img_mask.type(torch.float32)
    img_mask = torch.cat([img_mask] * 3, dim=0)

    salt_pepper = torch.randint(low=0, high=2, size=proper_size, device=input.device)
    salt_pepper = torch.cat([salt_pepper] * 3, dim=0).type(torch.float32)
    salt_pepper = minmax_scale(salt_pepper, max=1, min=0, new_max=1, new_min=-1)
    print(img_mask.shape)
    print(input.shape)
    print(noise_mask.shape)
    print(salt_pepper.shape)
    canvas = img_mask * input + noise_mask * salt_pepper
    
    return canvas

# color transformation
def gray_scale(input):
    out = tf.rgb_to_grayscale(input, num_output_channels=3)
    return out

def color_invert(input):
    input = minmax_scale(input, max=1, min=-1, new_max=1, new_min=0)
    out = minmax_scale(tf.invert(input), max=1, min=0, new_max=1, new_min=-1)
    return out

channel_set = [[0,2,1], [1,2,0], [1,0,2], [2,1,0], [2,0,1]]
def hue_rotation(input): # input shape = (3, H, W): tensor
    set_num = random.randrange(0,5)
    channels = channel_set[set_num]
    output = [input[channels[0:1]], input[channels[1:2]], input[channels[2:3]]]
    output = torch.cat(output, dim=0)
    return output

# geometric transformation
def x_flip(input):
    out = tf.hflip(input)
    return out

def right_rotation(input):
    out = tf.rotate(input, angle=90, expand=True)
    return out

def left_rotation(input):
    out = tf.rotate(input, angle=270, expand=True)
    return out

def horizontal_rotation(input):
    out = tf.rotate(input, angle=180, expand=True)
    return out

# occlusion
def rand_erasing(input):
    erase_value = [minmax_scale(random.random(), max=1, min=0, new_max=1, new_min=-1) for i in range(3)]
    erase = transforms.RandomErasing(p=1.0, value=erase_value)
    out = erase(input)
    return out

# bypass
def bypass(input):
    return input

# geometric transformation in Ada
def iso_scaling(input):
    height = input.shape[1]
    width = input.shape[2]
    scale_factor = np.tanh(np.random.randn())/4 + 1 # 0.75 ~ 1.25

    resized_height = int(height*scale_factor)
    resized_width = int(width*scale_factor)

    input = tf.resize(input, size=(resized_height, resized_width), interpolation=tf.InterpolationMode.BILINEAR)

    if scale_factor < 1:
        left_padding = int((width - resized_width)/2)
        right_padding = int(width - resized_width - left_padding)
        top_padding = int((height - resized_height)/2)
        bottom_padding = int(height - resized_height - top_padding)
        out = tf.pad(input, padding=[left_padding, top_padding, right_padding, bottom_padding], padding_mode='reflect')
    
    elif scale_factor >=1:
        out = tf.center_crop(input, output_size=[height, width])

    return out

def arbit_rotation(input):
    rotate_factor = np.tanh(np.random.randn()) * 180
    out = tf.rotate(input, angle=rotate_factor, expand=False, interpolation=tf.InterpolationMode.BILINEAR)
    return out

def aniso_scaling(input):
    height = input.shape[1]
    width = input.shape[2]
    scale_factor = np.tanh(np.random.randn())/4

    if scale_factor > 0:
        resized_width = int(width/(scale_factor + 1))
        padding_left = int((width - resized_width)/2)
        padding_right = width - (resized_width + padding_left)
        out = tf.pad(tf.resize(input, [height, resized_width], interpolation=tf.InterpolationMode.BILINEAR), padding=[padding_left, 0, padding_right, 0], padding_mode='reflect')
    elif scale_factor <=0:
        resized_height = int(height/(-scale_factor + 1))
        padding_top = int((height - resized_height)/2)
        padding_bottom = height - (resized_height + padding_top)
        out = tf.pad(tf.resize(input, [resized_height, width], interpolation=tf.InterpolationMode.BILINEAR), padding=[0, padding_top, 0, padding_bottom], padding_mode='reflect')

    return out

def translation_2d(input):
    height = input.shape[1]
    width = input.shape[2]
    x_trans = int(np.tanh(np.random.randn()) * (width/4))
    y_trans = int(np.tanh(np.random.randn()) * (height/4))

    if x_trans > 0 and y_trans > 0:
        padding_list = [x_trans, y_trans, 0, 0]
        input = input[:, :-y_trans, :-x_trans]
    elif x_trans < 0 and y_trans > 0:
        padding_list = [0, y_trans, -x_trans, 0]
        input = input[:, :-y_trans, -x_trans:]
    elif x_trans > 0 and y_trans < 0:
        padding_list = [x_trans, 0, 0, -y_trans]
        input = input[:, -y_trans:, :-x_trans]
    elif x_trans < 0 and y_trans < 0:
        padding_list = [0, 0, -x_trans, -y_trans]
        input = input[:, -y_trans:, -x_trans:]
    elif x_trans == 0 and y_trans > 0:
        padding_list = [0, y_trans, 0, 0]
        input = input[:, :-y_trans, :]
    elif x_trans == 0 and y_trans == 0:
        padding_list = [0, 0, 0, 0]
        input = input
    elif x_trans == 0 and y_trans < 0:
        padding_list = [0, 0, 0, -y_trans]
        input = input[:, -y_trans:, :]
    elif x_trans > 0 and y_trans == 0:
        padding_list = [x_trans, 0, 0, 0]
        input = input[:, :, :-x_trans]
    elif x_trans < 0 and y_trans == 0:
        padding_list = [0, 0, -x_trans, 0]
        input = input[:, :, -x_trans:]
    
    out = tf.pad(input, padding=padding_list, padding_mode='reflect')
    return out

# utility
def minmax_scale(input, max=255, min=0, new_max=1, new_min=-1): # input: tensor
    if max == None and min == None:
        max = input.max()
        min = input.min()

    return (input - min)/(max - min) * (new_max - new_min) + new_min

def numpy2tensor(input):
    return torch.from_numpy(input)

def open_image(input): # datatype = path
    bgr_image = cv.imread(input) # BGR, (H, W, C)
    rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB).astype(np.float32) # RGB, (H, W, C)
    rgb_image = np.transpose(rgb_image, (2, 0, 1)) # RGB (C, H, W)
    return rgb_image

# test augmentation
def test_aug(path, aug):
    img = aug(minmax_scale(numpy2tensor(open_image(path))))
    img = minmax_scale(img, max=1, min=-1, new_max=1, new_min=0)
    img = img.permute(1,2,0)
    plt.imshow(img)
    plt.show()

# random augmentation
# usage
"""
augmented_img = aug(img, [gau_noise, gau_blur, right_rotation])
"""
def aug(img, aug_list):
    aug_len = len(aug_list)
    aug_num = random.randrange(0, aug_len)

    aug_img = aug_list[aug_num](img)

    return aug_img

if __name__ == '__main__':
    test_aug('./032647.jpg', aniso_scaling)
