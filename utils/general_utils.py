#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
import torch.nn.functional as F
from PIL import Image
import os
import imageio
from tqdm import tqdm

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def pred_weights(gt_image, transient_model):
    width, height = gt_image.shape[1:]
    target_width = (width + 31) // 32 * 32
    target_height = (height + 31) // 32 * 32


    pad_width_left = (target_width - width) // 2
    pad_width_right = target_width - width - pad_width_left
    pad_height_top = (target_height - height) // 2
    pad_height_bottom = target_height - height - pad_height_top


    padded_image = F.pad(gt_image, 
                        (pad_height_top, pad_height_bottom, pad_width_left, pad_width_right), 
                        mode='constant', 
                        value=0)  # Here, using 0 for padding, could be others like 'reflect', 'replicate'

    weights = transient_model(padded_image.unsqueeze(0)).squeeze()

    rec_weights = weights[pad_width_left: pad_width_left + width, pad_height_top: pad_height_top + height]

    return rec_weights

def prep_img(img):
    to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
    return to8b(img.detach()).transpose(1,2,0)

def mask_image(gt_image, transient_model, threshold = 0.5):
    weights = pred_weights(gt_image, transient_model)
    mask_np = (weights>threshold).detach().cpu().numpy()
    mask_ = np.expand_dims(np.array(mask_np),2).repeat(3, axis=2)
    img_np = prep_img(gt_image)

    h,w = img_np.shape[:2]
    green = np.zeros([h, w, 3]) 
    green[:,:,1] = 255
    alpha = 0.6
    fuse_img = (1-alpha)*img_np + alpha*green
    fuse_img = mask_ * fuse_img + (1-mask_)*img_np

    return Image.fromarray(fuse_img.astype(np.uint8))


def make_gif(images, path_to_save, framerate=20, rate=10):
    
    os.system("rm -r /tmp_images")
    os.system("mkdir /tmp_images/")

    img_path = "/tmp_images/"

    for idx, img in tqdm(enumerate(images)):
        imageio.imwrite(os.path.join(img_path, '{0:05d}'.format(idx) + ".png"), img)

    os.system(f"ffmpeg -framerate {framerate} -i {img_path}/%05d.png -r {rate} -s 640x480  {path_to_save} -y")
    os.system("rm -r /tmp_images")


