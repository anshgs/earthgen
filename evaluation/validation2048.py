import math
import os
import random
import PIL
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, utils
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from patchify import patchify
import numpy as np
from tqdm.auto import tqdm
from numpy import pi, exp, sqrt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
import torchvision.transforms as T
import sys
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler,  DPMSolverSinglestepScheduler
from diffusers.optimization import get_scheduler
import matplotlib.pyplot as plt
from diffusers import StableDiffusionUpscalePipeline
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load",help="model weight folder", required=True)
parser.add_argument("-vr", "--validreg",help="validation image path (level 20)", required=True)
parser.add_argument("-or", "--outreg",help="output path (level 20)", required=True)
parser.add_argument("-vu", "--validurb",help="validation image path (urban level 20)", required=True)
parser.add_argument("-ou", "--outurb",help="output path (urban level 20)", required=True)
parser.add_argument("-v", "--vae",help="vae weight path", required=True)
args = parser.parse_args()

noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", subfolder="tokenizer"
)
text_encoder = CLIPTextModel.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", subfolder="text_encoder"
).cuda()

vae = AutoencoderKL.from_pretrained(
    args.vae, subfolder="vae"
).cuda()

scheduler = DPMSolverSinglestepScheduler.from_config(pipeline.scheduler.config)

low_res_scheduler = DDPMScheduler.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", subfolder="low_res_scheduler"
)

def get_tokenized_caption(caption):
    captions = [caption]
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

def get_views(panorama_height, panorama_width, window_size=128, stride=64):
    panorama_height //= 4
    panorama_width //= 4
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views


def gaussian_weights(tile_width, tile_height, nbatches=1):
    
    latent_width = tile_width // 4
    latent_height = tile_height // 4

    var = 0.01
    midpoint = (latent_width - 1) / 2 
    x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
    midpoint = latent_height / 2
    y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

    weights = np.outer(y_probs, x_probs)
    return torch.tile(torch.tensor(weights), (nbatches, 4, 1, 1))

def crop_center(im):
    width, height = im.size 
    crop = 512

    left = (width - crop)/2
    top = (height - crop)/2
    right = (width + crop)/2
    bottom = (height + crop)/2
    ret = im.crop((left, top, right, bottom))
    return ret


tile_dim = 512
vis_dim = 512
out_dim = 2048
batch_size = 512

transform = T.ToPILImage()

    
def get_image_mixture(lrim, unet, bs, timesteps_k, latent_save_path, guidance_scale):
    unet.eval()
    latents = torch.randn((1, 4, out_dim//4, out_dim//4)).to('cuda')
    prompt_embeds = text_encoder(torch.unsqueeze(get_tokenized_caption("satellite photo")[0], dim=0).to('cuda'))[0]
    neg_embeds = text_encoder(torch.unsqueeze(get_tokenized_caption("blurry, lowres, low quality")[0], dim=0).to('cuda'))[0]
    scheduler.set_timesteps(timesteps_k, device='cpu')
    timesteps = scheduler.timesteps

    views = get_views(out_dim, out_dim)
    count = torch.zeros_like(latents).cpu()
    value = torch.zeros_like(latents).cpu()
    weights = gaussian_weights(tile_dim, tile_dim)
    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps,desc="timesteps", position=0,leave=True)):
           
            count.zero_()
            value.zero_()
            stacked = []
            locs = []
            curcount = 0
            for j, view in enumerate(tqdm(views, desc="views", position=1,leave=True)):
                h_start, h_end, w_start, w_end = view
                latent_view = latents[:, :, h_start:h_end, w_start:w_end]
                
                latent_view = scheduler.scale_model_input(latent_view, t)
                image_lr = torch.unsqueeze(lrim[:, h_start:h_end, w_start:w_end], dim=0)
                latent_view = torch.cat([latent_view, image_lr], dim=1)
                stacked.append(latent_view)
                locs.append(view)
                curcount+=1
                if curcount == bs or j == len(views)-1:
                    prompt_embeds_batched = torch.cat([prompt_embeds]*len(stacked))
                    neg_embeds_batched = torch.cat([neg_embeds]*len(stacked))
                    noise_pred = unet(torch.cat(stacked), torch.tensor([t]*curcount), prompt_embeds_batched, class_labels=torch.tensor([20]*curcount, dtype=torch.long, device='cuda'), return_dict=False).cpu()
                    noise_pred_neg = unet(torch.cat(stacked), torch.tensor([t]*curcount), neg_embeds_batched, class_labels=torch.tensor([20]*curcount, dtype=torch.long, device='cuda'), return_dict=False).cpu()
                    noise_pred = noise_pred + guidance_scale * (noise_pred - noise_pred_neg)
                    for k in range(len(stacked)):
                        
                        b_h_start, b_h_end, b_w_start, b_w_end = locs[k]
                        value[:, :, b_h_start:b_h_end, b_w_start:b_w_end] += noise_pred[k,:,:,:]*weights
                        count[:, :, b_h_start:b_h_end, b_w_start:b_w_end] += weights
                        
                    curcount = 0
                    stacked = []
                    locs = []
                    
            noise_pred = torch.where(count > 0, value / count, value)
            
            
            
            latents = scheduler.step(noise_pred, t, latents.cpu(), return_dict=False)[0].cuda()
            
        torch.save(latents, latent_save_path)


def blend_v(a, b, blend_extent):
    blend_extent = min(a.shape[2], b.shape[2], blend_extent)
    for y in range(blend_extent):
        b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
    return b

def blend_h(a, b, blend_extent):
    blend_extent = min(a.shape[3], b.shape[3], blend_extent)
    for x in range(blend_extent):
        b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
    return b


def tiled_decode(z, vae):
    vae.tile_latent_min_size=512
    vae.tile_sample_min_size=2048
    vae.tile_overlap_factor = 0.25
    overlap_size = int(vae.tile_latent_min_size * (1 - vae.tile_overlap_factor))
    blend_extent = int(vae.tile_sample_min_size * vae.tile_overlap_factor)
    row_limit = vae.tile_sample_min_size - blend_extent
    print(overlap_size, blend_extent, row_limit)
    rows = []
    for i in range(0, z.shape[2], overlap_size):
        row = []
        for j in range(0, z.shape[3], overlap_size):
            tile = z[:, :, i : i + vae.tile_latent_min_size, j : j + vae.tile_latent_min_size]
            tile = tile.cuda()
            tile = vae.post_quant_conv(tile)
            decoded = vae.decoder(tile).cpu()
            row.append(decoded)
        rows.append(row)
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend_v(rows[i - 1][j].cuda(), tile.cuda(), blend_extent).cpu()
            if j > 0:
                tile = blend_h(row[j - 1].cuda(), tile.cuda(), blend_extent).cpu()
            result_row.append(tile[:, :, :row_limit, :row_limit])
        result_rows.append(torch.cat(result_row, dim=3))

    dec = torch.cat(result_rows, dim=2)
    return dec


def get_out_image(latent_save_path, image_save_path, bs):
    latents_cpu = torch.load(latent_save_path).cpu()
    with torch.no_grad():
        result = tiled_decode(latents_cpu/vae.config.scaling_factor, vae)

    image = result[0]
    transform = T.ToPILImage()
    final_im = transform(torch.clamp((image+1)/2, min=0.0, max=1.0))
    final_im.save(image_save_path)

def generate_next_zoom(low_res, unet, batch_size, timesteps, latent_save_path, image_save_path, guidance_scale):
    get_image_mixture(train_transforms(low_res).cuda(), unet, batch_size, timesteps, latent_save_path, guidance_scale)

import os
import shutil

validation_path = args.validreg
out_path = args.outreg

if os.path.exists(out_path):
    shutil.rmtree(out_path)
os.makedirs(out_path)
    

for dir in os.listdir(validation_path):
    cat_dir = os.path.join(validation_path, dir)
    cat_dir_new = os.path.join(out_path, dir)
    os.makedirs(cat_dir_new)
    png_path = os.path.join(cat_dir, "10.png")
    jpg_path = os.path.join(cat_dir, "10.jpg")
    if os.path.exists(png_path):
        low_res_path = png_path
    else:
        low_res_path = jpg_path


    low_res = Image.open(low_res_path)
    low_res.save(os.path.join(cat_dir_new, "10_gt.png"))
    low_res = crop_center(low_res)
    unet = torch.nn.DataParallel(UNet2DConditionModel.from_pretrained(
                os.path.join(args.load, '10to12'), subfolder="unet"
            ), device_ids=[0,1,2,3], dim=0).cuda()
    path_12 = os.path.join(cat_dir_new, "10_gt_12.png")
    generate_next_zoom(low_res, unet, batch_size, 50, 'latents_10.pt', path_12, 5)

    low_res = Image.open(path_12)
    low_res = crop_center(low_res)
    unet = torch.nn.DataParallel(UNet2DConditionModel.from_pretrained(
                os.path.join(args.load, '12to14'), subfolder="unet"
            ), device_ids=[0,1,2,3], dim=0).cuda()
    path_14 = os.path.join(cat_dir_new, "10_gt_14.png")
    generate_next_zoom(low_res, unet, batch_size, 50, 'latents_12.pt', path_14, 2)



    low_res = Image.open(path_14)
    low_res = crop_center(low_res)
    unet = torch.nn.DataParallel(UNet2DConditionModel.from_pretrained(
                os.path.join(args.load, '14to16'), subfolder="unet"
            ), device_ids=[0,1,2,3], dim=0).cuda()
    path_16 = os.path.join(cat_dir_new, "10_gt_16.png")
    generate_next_zoom(low_res, unet, batch_size, 50, 'latents_14.pt', path_16, 3)

    low_res = Image.open(path_16)
    low_res = crop_center(low_res)
    unet = torch.nn.DataParallel(UNet2DConditionModel.from_pretrained(
                os.path.join(args.load, '16to18'), subfolder="unet"
            ), device_ids=[0,1,2,3], dim=0).cuda()
    path_18 = os.path.join(cat_dir_new, "10_gt_18.png")
    generate_next_zoom(low_res, unet, batch_size, 50, 'latents_16.pt', path_18, 3)

    low_res = Image.open(path_18)
    low_res = crop_center(low_res)
    unet = torch.nn.DataParallel(UNet2DConditionModel.from_pretrained(
                os.path.join(args.load, '18to20'), subfolder="unet"
            ), device_ids=[0,1,2,3], dim=0).cuda()
    path_20 = os.path.join(cat_dir_new, "10_gt_20.png")
    generate_next_zoom(low_res, unet, batch_size, 50, 'latents_18.pt', path_20, 4)

    low_res = Image.open(path_20)
    low_res = crop_center(low_res)

validation_path = args.validurb
out_path = args.outurb

if os.path.exists(out_path):
    shutil.rmtree(out_path)
os.makedirs(out_path)
    

for dir in os.listdir(validation_path):
    cat_dir = os.path.join(validation_path, dir)
    cat_dir_new = os.path.join(out_path, dir)
    os.makedirs(cat_dir_new)
    png_path = os.path.join(cat_dir, "10.png")
    jpg_path = os.path.join(cat_dir, "10.jpg")
    if os.path.exists(png_path):
        low_res_path = png_path
    else:
        low_res_path = jpg_path


    low_res = Image.open(low_res_path)
    low_res.save(os.path.join(cat_dir_new, "10_gt.png"))
    low_res = crop_center(low_res)
    unet = torch.nn.DataParallel(UNet2DConditionModel.from_pretrained(
                os.path.join(args.load, '10to12'), subfolder="unet"
            ), device_ids=[0,1,2,3], dim=0).cuda()
    path_12 = os.path.join(cat_dir_new, "10_gt_12.png")
    generate_next_zoom(low_res, unet, batch_size, 50, 'latents_10.pt', path_12, 5)

    low_res = Image.open(path_12)
    low_res = crop_center(low_res)
    unet = torch.nn.DataParallel(UNet2DConditionModel.from_pretrained(
                os.path.join(args.load, '12to14'), subfolder="unet"
            ), device_ids=[0,1,2,3], dim=0).cuda()
    path_14 = os.path.join(cat_dir_new, "10_gt_14.png")
    generate_next_zoom(low_res, unet, batch_size, 50, 'latents_12.pt', path_14, 2)



    low_res = Image.open(path_14)
    low_res = crop_center(low_res)
    unet = torch.nn.DataParallel(UNet2DConditionModel.from_pretrained(
                os.path.join(args.load, '14to16'), subfolder="unet"
            ), device_ids=[0,1,2,3], dim=0).cuda()
    path_16 = os.path.join(cat_dir_new, "10_gt_16.png")
    generate_next_zoom(low_res, unet, batch_size, 50, 'latents_14.pt', path_16, 3)

    low_res = Image.open(path_16)
    low_res = crop_center(low_res)
    unet = torch.nn.DataParallel(UNet2DConditionModel.from_pretrained(
                os.path.join(args.load, '16to18'), subfolder="unet"
            ), device_ids=[0,1,2,3], dim=0).cuda()
    path_18 = os.path.join(cat_dir_new, "10_gt_18.png")
    generate_next_zoom(low_res, unet, batch_size, 50, 'latents_16.pt', path_18, 3)

    low_res = Image.open(path_18)
    low_res = crop_center(low_res)
    unet = torch.nn.DataParallel(UNet2DConditionModel.from_pretrained(
                os.path.join(args.load, '18to20'), subfolder="unet"
            ), device_ids=[0,1,2,3], dim=0).cuda()
    path_20 = os.path.join(cat_dir_new, "10_gt_20.png")
    generate_next_zoom(low_res, unet, batch_size, 50, 'latents_18.pt', path_20, 4)

    low_res = Image.open(path_20)
    low_res = crop_center(low_res)
