from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

from PIL import Image
import numpy as np
import cv2
import tqdm
import argparse

from annotator.uniformer.mmseg.core.evaluation import get_palette
from constant import *

palette = get_palette('ade')
palette = np.array(palette)

parser = argparse.ArgumentParser(description="DifFSS")
parser.add_argument('--st', type=int, default=0)
parser.add_argument('--end', type=int, default=2)
parser.add_argument('--imgdir', type=str, default='/data/user6/coco/')
parser.add_argument('--maskdir', type=str, default='/data/user6/coco/annotations/')
parser.add_argument('--dstdir', type=str, default='/data/user6/justtest/')
parser.add_argument('--list', type=str, default='./list/coco_all.txt')
parser.add_argument('--dataset', type=str, choices=['pascal', 'coco', 'fss'], default='fss')
parser.add_argument('--guidance', type=str, choices=['seg', 'hed', 'scribble'], default='seg')
parser.add_argument('--save_control', type=int, choices=[0, 1], default=1)
args = parser.parse_args() 

print(args)

st = args.st
end = args.end

img_basedir = args.imgdir
mask_basedir = args.maskdir
force_random = True
shot = 4
img_basedir = args.imgdir
mask_basedir = args.maskdir

detect_size = 512
force_random = True
shot = 4

target_dir = args.dstdir

a_prompt =  'best quality, extremely detailed'
n_prompt =  'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, artwork'
num_samples = 1
detect_resolution = 512
image_resolution = 512
ddim_steps = 2
guess_mode = False
strength = 1
scale = 9
seed = -1

category_name = eval(f'{args.dataset}_name()')


def seg_init():
    global model, ddim_sampler, category_mapper
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('./models/control_sd15_seg.pth', location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    category_mapper = eval(f'{args.dataset}_mapper()')
    return model, ddim_sampler, None

def hed_init():
    global model, ddim_sampler, apply_hed
    from annotator.hed import HEDdetector
    apply_hed = HEDdetector()
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('./models/control_sd15_hed.pth', location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    return model, ddim_sampler, apply_hed

def scribble_init():
    global model, ddim_sampler, apply_hed
    from annotator.hed import HEDdetector
    apply_hed = HEDdetector()
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('./models/control_sd15_scribble.pth', location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    return model, ddim_sampler, apply_hed

def pascal_list(path=args.list):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    with open(path, 'r') as f:
        fold_n_metadata = f.read().split('\n')[:-1]
    fold_n_metadata = [data[11:22] for data in fold_n_metadata]
    return fold_n_metadata

def coco_list(path=args.list):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    if not os.path.exists(target_dir + 'train2014'):
        os.mkdir(target_dir + 'train2014')
    if not os.path.exists(target_dir + 'val2014'):
        os.mkdir(target_dir + 'val2014')

    with open(path, 'r') as f:
        fold_n_metadata = f.read().split('\n')[:-1]
    fold_n_metadata = [data.split(' ')[0].split('.')[0] for data in fold_n_metadata]
    return fold_n_metadata

def fss_list(path='./list/fss1k_all.txt'):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    with open(path, 'r') as f:
        fold_n_metadata = f.read().split('\n')[:-1]
    fss_list = [data for data in fold_n_metadata]
    for cls_name in fss_list:
        if not os.path.exists(target_dir + cls_name):
            os.mkdir(target_dir + cls_name)
    new_list = [f'{cls_name}/{index}' for cls_name in fold_n_metadata for index in range(1,11)]
    return new_list


def seg_control(img, mask, fg_cls, ):
    seg_mask = palette[(mask == fg_cls) * category_mapper[fg_cls]]
    H, W, C = resize_image(img, image_resolution).shape
    print('curr size:' + f'{H},{W}' )

    detected_map = cv2.resize(seg_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    print(control.shape)
    return control, detected_map.astype( np.uint8 )

def hed_control(img, mask, fg_cls, ):
    raw_detected_map = apply_hed(resize_image(img, detect_resolution))
    mask = cv2.resize(mask, (raw_detected_map.shape[1],raw_detected_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    fgmask = np.array(mask==fg_cls,dtype=int).astype('uint8') 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (14,14))
    dilated = cv2.dilate(fgmask, kernel, 10)
    filtered_detected_map = np.multiply(raw_detected_map, dilated)

    detected_map = HWC3(filtered_detected_map)

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    return control, detected_map.astype( np.uint8 )

def scribble_control(img, mask, fg_cls, ):
    raw_detected_map = apply_hed(resize_image(img, detect_resolution))
    mask = cv2.resize(mask, (raw_detected_map.shape[1],raw_detected_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    fgmask = np.array(mask==fg_cls,dtype=int).astype('uint8') 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (14,14))
    dilated = cv2.dilate(fgmask, kernel, 10)
    input_mask = resize_image(HWC3(dilated * raw_detected_map), image_resolution)

    detected_map = np.zeros_like(input_mask, dtype=np.uint8)
    detected_map[np.min(input_mask, axis=2) > 127] = 255

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    return control, detected_map.astype( np.uint8 )


def generate(img_name, target_dir):
    global seed
    img = Image.open(img_basedir +  img_name + '.jpg')
    img = np.array(img)
    img = HWC3(img)
    H, W, C = resize_image(img, image_resolution).shape
    mask = cv2.imread(mask_basedir + img_name + '.png', cv2.IMREAD_GRAYSCALE)
    mask = np.array(mask)

    #  fss mask binarize
    if args.dataset == 'fss':
        filterr = 100
        mask = np.array(mask >= filterr,dtype=np.uint8)
        
    for fg_cls in np.unique(mask):
        if fg_cls in [0, 255]: continue
        if fg_cls not in list(range(0,21)): continue
        control, detected_map = eval(f'{args.guidance}_control')(img, mask, fg_cls)
        prompt = f'a real shot photo of {category_name[fg_cls]},'
        
        for shot_index in range(shot):
            if seed == -1 or force_random:
                seed = random.randint(0, 65535)
            seed_everything(seed)
            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)
            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)
            if config.save_memory:
                model.low_vram_shift(is_diffusing=True)
            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=0.0,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)
            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)
            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            results = [x_samples[i] for i in range(num_samples)]
            cv2.imwrite(target_dir + img_name + f'_{fg_cls}_{shot_index}.jpg', cv2.cvtColor(results[0], cv2.COLOR_BGR2RGB))
        
        if args.save_control == 1:
            cv2.imwrite(target_dir + img_name + f'_{fg_cls}_control.jpg', cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))

if __name__ == '__main__':
    eval(f'{args.guidance}_init')()
    tasklist = eval(f'{args.dataset}_list')()
    end = len(tasklist) if end == -1 else end
    for i in tqdm.tqdm(range(st,end)):
        generate(tasklist[i], target_dir=target_dir)

