from __future__ import absolute_import

import torch
import torch.nn as nn
import numpy as np
import scipy.misc
from PIL import Image

from .misc import *
import copy, math
import torch.nn.functional as F

def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def im_to_torch_no_normalize(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    # img = to_torch(img).float()
    return img

def load_image(img_path):
    # H x W x C => C x H x W
    img = np.array(Image.open(img_path))
    return im_to_torch_no_normalize(img)

def load_attention_heatmap(heatmap_path) :
    heatmap = np.array(Image.open(heatmap_path))
    heatmap = np.expand_dims(heatmap, -1) # H x W -> H x W X 1
    return im_to_torch(heatmap) #  H x W X 1 -> 1 x H x W

def load_mask(mask_path):
    # H x W x C => C x H x W
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 2 :
        mask = np.expand_dims(mask, -1) # H x W x C

    # deal with two kind of mask here
    # 1. C = 1: place not 0 should be value 1 (valid mask)
    # 2. C = 3: all black mask
    _mask = np.zeros((mask.shape[0], mask.shape[1], 1))
    if mask.shape[2] == 1 :
        non_zero = mask.nonzero() ## can not transformed into array !!!!
        _mask[non_zero] = 1
    elif mask.shape[2] == 3 :
        pass
    
    return im_to_torch(_mask) # need to normalize

def load_depth(depth_path):
    max_depth = 10000 # actual number is 9xxxx
    depth = np.load(depth_path)
    # project into image space for further resize 
    depth = depth / 10000 * 255
    depth = np.asarray(depth, dtype = np.uint8)
    depth = np.transpose(depth, (2, 0, 1)) # C*H*W
    return depth

def resize(img, owidth, oheight):  # CxHxW -> HxWxC
    img = to_numpy(img)
    if img.ndim == 2:
        pass
    elif img.ndim == 3 :
        img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:
            img = np.squeeze(img, -1)

    img = np.asarray(img, dtype = np.uint8)

    ## Future work

    # scipy.misc.imresize ?

    # img = F.interpolate()

    ## Use F.interpolate to replace
    img = np.array(Image.fromarray(img).resize((owidth, oheight)))

    if img.ndim == 2:
        img = np.expand_dims(img, -1)

    img = im_to_torch(img)
    # print('%f %f' % (img.min(), img.max()))
    return img


# =============================================================================
# Helpful functions generating groundtruth labelmap
# =============================================================================

def gaussian(shape=(7,7),sigma=1):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    return to_torch(h).float()

def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img), 0

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)


    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return to_torch(img), 1

# =============================================================================
# Helpful display functions
# =============================================================================

def gauss(x, a, b, c, d=0):
    return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

def color_heatmap(x, mode):
    x = to_numpy(x) # 256x256
    color = np.zeros((x.shape[0],x.shape[1],3))
    if mode == 'gt' :
        gt_pos = (x != 0).nonzero()
        # print(gt_pos)
    elif mode == 'pred' :
        x = x * 255 # because x is normalized before
        gt_pos = (x >= 0.5).nonzero() # have to exceed threshold
    pos_0, pos_1 = gt_pos
    color[pos_0, pos_1, 0] = 1
    color = (color * 255).astype(np.uint8)
    return color


def imshow(img):
    npimg = im_to_numpy(img*255).astype(np.uint8)
    plt.imshow(npimg)
    plt.axis('off')

def show_joints(img, pts):
    imshow(img)

    for i in range(pts.size(0)):
        if pts[i, 2] > 0:
            plt.plot(pts[i, 0], pts[i, 1], 'yo')
    plt.axis('off')

def show_sample(inputs, target):
    num_sample = inputs.size(0)
    num_joints = target.size(1)
    height = target.size(2)
    width = target.size(3)

    for n in range(num_sample):
        inp = resize(inputs[n], width, height)
        out = inp
        for p in range(num_joints):
            tgt = inp*0.5 + color_heatmap(target[n,p,:,:])*0.5
            out = torch.cat((out, tgt), 2)

        imshow(out)
        plt.show()

def sample_test(inp):
    inp = to_numpy(inp[0] * 255)
    img = np.zeros((inp.shape[1], inp.shape[2], inp.shape[0]))
    for i in range(3):
        img[:, :, i] = inp[i, :, :]
    img = np.asarray(img, np.uint8)
    return img

def sample_with_heatmap(inp, out, mode, num_rows=2, parts_to_show=None):
    inp = to_numpy(inp * 255)
    out = to_numpy(out)

    img = np.zeros((inp.shape[1], inp.shape[2], inp.shape[0]))
    for i in range(3):
        img[:, :, i] = inp[i, :, :]

    img = np.asarray(img, np.uint8)
    # print(img.shape) # 256 256 3

    size = img.shape[0] # 256

    full_img = np.zeros((img.shape[0], size * 2, 3), np.uint8) # 256 512 3
    full_img[:img.shape[0], :img.shape[1]] = img

    inp_small = np.array(Image.fromarray(img).resize((size, size)))

    part_idx = 0
    out_resized = np.array(Image.fromarray(out[part_idx]).resize((size, size)))
    out_resized = out_resized.astype(float)/255
    
    # print(out_resized.shape) # 256 256

    color_hm = color_heatmap(out_resized, mode)
    out_img = inp_small.copy() * .3
    out_img += color_hm * .7
    # out_img = color_hm.copy()

    # full_img[row_offset:row_offset + size, col_offset:col_offset + size] = out_img
    full_img[0: size, 256:256 + size] = out_img

    return full_img

def batch_with_heatmap(inputs, outputs, mode, mean=torch.Tensor([0.5, 0.5, 0.5]).cuda(), num_rows=2, parts_to_show=None):
    batch_img = []
    for n in range(min(inputs.size(0), 4)):
        inp = inputs[n]
        batch_img.append(
            sample_with_heatmap(inp.clamp(0, 1), outputs[n], mode, num_rows=num_rows, parts_to_show=parts_to_show)
        )
    return np.concatenate(batch_img)


'''
Below is only for relabel
'''

def color_heatmap_relabel(x, mode):
    x = to_numpy(x) # 256x256
    color = np.zeros((x.shape[0],x.shape[1]))
    if mode == 'gt' :
        gt_pos = (x != 0).nonzero()
    elif mode == 'pred' :
        x = x * 255 # because x is normalized before
        gt_pos = (x >= 0.5).nonzero() # have to exceed threshold
    pos_0, pos_1 = gt_pos
    color[pos_0, pos_1] = 1
    color = (color * 255).astype(np.uint8)
    return color

### FINAL
def relabel_with_heatmap(inp, out, mode, area_to_detect=None, num_rows=2):
    '''
    area_to_detect : [[x_min, y_min, x_max, y_max]
                        ...] for 640 x 480 image
    '''
    inp = to_numpy(inp * 255) # [3, 256, 256]
    out = to_numpy(out) # [1, 64, 64]

    img = np.zeros((inp.shape[1], inp.shape[2], inp.shape[0]))
    for i in range(3):
        img[:, :, i] = inp[i, :, :]

    img = np.asarray(img, np.uint8)
    # print(img.shape) # 256 256 3

    size = img.shape[0] # 256

    full_img = np.zeros((640, 480, 3), np.uint8) # 256 512 3

    inp_small = np.array(Image.fromarray(img).resize((size, size)))

    part_idx = 0
    out_resized = np.array(Image.fromarray(out[part_idx]).resize((size, size))) # [256, 256]
    out_resized = out_resized.astype(float)
    
    # print(out_resized.shape) # 256 256
    # print(out_resized[out_resized > 0.5])
    
    '''
    # 2020.7.1
    ## use area_to_detect to occlude other area
    if area_to_detect is not None :
        out_resized_area = np.zeros((size, size))
        for i in range(len(area_to_detect)):
            x_min, y_min, x_max, y_max = area_to_detect[i]
            # x_min = 640 - x_max
            # y_min = 480 - y_max
            # x_max = 640 - x_min
            # y_max = 480 - y_min
            
            # from (640 480) -> (256 256)
            x_min = math.floor(x_min / 640 * 255)
            y_min = math.floor(y_min / 480 * 255)
            x_max = math.ceil(x_max / 640 * 255)
            y_max = math.ceil(y_max / 480 * 255)

            # direction is differnt. one is x, y, another is y, x
            out_resized_area[y_min:y_max, x_min:x_max] = out_resized[y_min:y_max, x_min:x_max]

        out_resized = copy.deepcopy(out_resized_area)
    '''



    color_hm_relabel = color_heatmap_relabel(out_resized, mode)
    output_mask = color_hm_relabel.copy()

    # old 
    # color_hm = color_heatmap(out_resized, mode)
    # out_img = inp_small.copy() * .99
    # out_img += color_hm * .7

    # new
    inp_small = np.asarray(inp_small, dtype = np.uint8) # [256, 256, 3] # [256, 256]
    out_img = eval_image_plus_mask(inp_small, out_resized, mode)
    out_img = np.asarray(out_img, dtype = np.uint8)

    full_img = Image.fromarray(out_img).resize((640, 480))
    output_mask = Image.fromarray(output_mask).resize((640, 480))

        
    full_img = np.array(full_img)
    full_img = np.asarray(full_img, np.uint8)

    # output_mask = np.array(output_mask)
    # output_mask = np.asarray(output_mask, np.uint8)

    return full_img, output_mask

def relabel_heatmap(inputs, outputs, mode, mean=torch.Tensor([0.5, 0.5, 0.5]).cuda(), area_to_detect=None, num_rows=2):
    inp = inputs[0]
    return relabel_with_heatmap(inp.clamp(0, 1), outputs[0], mode, area_to_detect=area_to_detect, num_rows=num_rows)

##############
# Eval
##############

def eval_image_plus_mask(img, mask, mode):
    mask = to_numpy(mask) # 256x256
    if mode == 'pred' :
        gt_pos = (mask >= 0.5).nonzero() # checked
    elif mode == 'gt' :
        gt_pos = (mask >= 0.5).nonzero()
        # x = x * 255 # because x is normalized before
        # gt_pos = (x >= 0.5).nonzero() # have to exceed threshold
    pos_0, pos_1 = gt_pos
    img[pos_0, pos_1] = (255, 0, 0)
    return img

def eval_heatmap(inp, out):
    '''
    Input :
        inp (input image) : [H, W, 3]
    '''
    # inp [480, 640, 3] uint8
    # out [480, 640] uint8

    # print(inp)
    # print(inp.shape)

    # print(out)
    # print(out.shape)
    inp = np.array(Image.fromarray(inp).resize((256, 256))) # [256, 256, 3]
    out = np.array(Image.fromarray(out).resize((256, 256))) # [256, 256]

    
    img = np.asarray(inp, np.uint8)
    size = img.shape[0] # 256

    # full_img = np.zeros((640, 480, 3), np.uint8)
    inp_small = img

    out_resized = out.astype(float)/255

    '''
    inp_small [256, 256, 3] np.uint8
    out_resized [256, 256] float
    '''
    out_img = eval_image_plus_mask(inp_small, out_resized, 'pred')
    out_img = np.asarray(out_img, dtype = np.uint8)

    full_img = Image.fromarray(out_img).resize((640, 480))

    full_img = np.array(full_img)
    full_img = np.asarray(full_img, np.uint8)

    return full_img

#########
# 2020.7.6
# for faster rcnn bbox crop
def faster_rcnn_crop(output, x_len, y_len):
    output = np.array(output[0]) # [32, 32]
    resized_output = np.array(Image.fromarray(output).resize((x_len, y_len)))
    resized_output = to_torch(resized_output).float()
    return resized_output