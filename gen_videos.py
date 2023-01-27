### My custom file for generating videos, given different parameters for Matrix transformation


import copy
import numpy as np
from torchvision.transforms import ToPILImage, ToTensor
import matplotlib.pyplot as plt
import os
import torch

from util import renormalize
from typing import List
from tqdm import tqdm
from util import patch_util


from util import patch_util
from tqdm import tqdm

import einops

from training.networks_stylegan3 import Generator
import pickle


def test_create_random_grid_4k(G_base, patches, ws_list:List[torch.Tensor], grid_size:int = 4, random_state:int = 19, full_size:int = 4096, fs:int = None, verbose:bool = False) -> List[torch.Tensor]:
    items = []
    for i in tqdm(range(grid_size*grid_size), desc='Generating images 4k', disable=not verbose):        
        full = torch.zeros([1, 3, full_size, full_size])
        
        for bbox, transform in patches:
            #G_base.scale_mapping_kwargs = None
            if fs is not None:
                # change [0, 0] and [1, 1] indexes new scale, instead of 0.25
                transform[0, 0] = fs
                transform[1, 1] = fs
            img = patch_util.scale_condition_wrapper(G_base, ws_list[i], transform[None].cuda(), noise_mode='const', force_fp32=True)
            full[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]] = img
        img = renormalize.as_tensor(full[0], target='pt')
        items.append(img)
    return items
    
def create_patch_info(shift_start, shift_stop, scale_start, scale_stop, n):
    import numpy as np # Two
    values = list(np.linspace(start=shift_start, stop=shift_stop, num=n))    
    list_scale = list(np.linspace(start=scale_start, stop=scale_stop, num=n))   

    # actually construct the lists
    list_tx_ty = []
    for v in values:
        list_tx_ty.append((v, v))

    scales = []
    for w in list_scale:
        scales.append((w, w))

    return list_tx_ty, scales

def create_patches(patch, list_tx_ty, scales, add_extra_patches:bool = False):
    patches = []
    if add_extra_patches:
        patches.append(copy.deepcopy(patch))
    for i in range(len(list_tx_ty)):
        cp = copy.deepcopy(patch)
        cp[1][0][2] = list_tx_ty[i][0]
        cp[1][1][2] = list_tx_ty[i][1]
        # set scales
        cp[1][0][0] = scales[i][0]
        cp[1][1][1] = scales[i][1]

        patches.append(cp)
    if add_extra_patches:
        patches.append(copy.deepcopy(patch))
    return patches
from PIL import ImageFont, ImageDraw

from grids import create_random_grid_4k

# myFont = ImageFont.truetype('FreeMono.ttf', 20)
# for sel in selected:
#     frames = []
#     for i in range(len(patches)):
#         rez = test_create_random_grid_4k(G_base, [patches[i]], [ws_list[sel]], grid_size=1, random_state=19, full_size=1024, fs=None, verbose=False)
#         img  = ToPILImage()(rez)
#         I1 = ImageDraw.Draw(img)
#         I1.text((10, 10), f"sx {patches[i][1][0][0]} sy {patches[i][1][1][1]} tx {patches[i][1][0][2]} ty {patches[i][1][1][2]}", font=myFont, fill=(255, 255, 255))
#         frames.append(img)

def draw_text(fr, curr_patch):
    draw_text = f"sx {curr_patch[1][0][0]} sy {curr_patch[1][1][1]} tx {curr_patch[1][0][2]} ty {curr_patch[1][1][2]}"
    myFont = ImageFont.truetype('FreeMono.ttf', 20)
    img  = ToPILImage()(fr)
    I1 = ImageDraw.Draw(img)
    I1.text((10, 10), draw_text, font=myFont, fill=(255, 255, 255))
    return ToTensor()(img)

def generate_frames(G, patches, ws, write_text_on_images:bool = False):
    frames = []
    myFont = ImageFont.truetype('FreeMono.ttf', 20)
    for i in range(len(patches)):
        #frame = G_base.synthesis(patches[i], noise_mode='const')
        #frame = (frame + 1) / 2
        frame = create_random_grid_4k(G, [ws], [patches[i]], grid_size=1, random_state=19, full_size=1024, fs=None, verbose=False)
        
        if not write_text_on_images:
            frames.append(frame if isinstance(frame, torch.Tensor) else frame[0])
        else:
            frames.append(torch.stack([draw_text(f, p) for f, p in zip(frame, patches)], dim=0))
    return frames

import torchvision
import imageio
def write_frames_to_file(frames, path, format:str):
    fpath = path[:path.rfind('/')]
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    if not (format == 'jpg' or format == 'png' or format == 'mp4' or format == 'gif'):
        print(f"Unknown format {format}")
        return
    #print(f"Saving {len(frames)} frames to {path} as {format} files")
    
    if format == 'jpg' or format == 'png':
        if isinstance(frames, list) and len(frames) > 1:
            for i in range(len(frames)):
                torchvision.utils.save_image(frames[i], f"{path}_{i}.{format}")
        else:
            torchvision.utils.save_image(frames if isinstance(frames, torch.Tensor) else frames[0], f"{path}.{format}")
    elif format == 'mp4':
        with imageio.v2.get_writer(f"{path}.{format}", fps=1, format='FFMPEG', mode='I') as w: # outputs{extra_path}{fname}result_{sel}.mp4
            for im in frames:
                # if tensor, convert to PIL 
                if isinstance(im, torch.Tensor):
                    im = ToPILImage()(im)
                w.append_data(np.array(im))
    elif format == 'gif':
        imageio.mimsave(f"{path}.gif", frames, fps=1)

def load_G(G_path:str = 'training-runs/SS750/00010-stylegan3-r-LR-gpus4-batch32-gamma2/network-snapshot-000000.pkl'):
    with open(G_path, 'rb') as f:
        G_base: Generator = pickle.load(f)['G_ema'].cuda()
    return G_base

def load_ws(ws_path:str = 'ws.pkl'):
    with open(ws_path, 'rb') as f:
        ws_list = pickle.load(f)
    return ws_list

def generate_different_images(selected, ws_list, G_base, output_resolution=1024, output_format="mp4",
                             shift_start=-0.5, shift_stop=0.5, scale_start=0.5, scale_stop=1.5, n=8, extra:str="r000000/", write_text_on_images:bool = False):
    patches = patch_util.generate_full_from_patches(output_resolution, G_base.img_resolution)
    folder_location = f"outputs/{extra}s{shift_start}_s{shift_stop}_sc{scale_start}_sc{scale_stop}_n{n}"
    adding_extra_patches = True if output_format == 'mp4' or output_format == 'gif' else False
    print(f"Saving to {folder_location} and {adding_extra_patches=}")
    for i in tqdm(selected, desc="Generating different images", leave=False):
        # output includes generator steps, 
        output_folder_and_file = f"{folder_location}/result_{i}"
        list_tx_ty, scales = create_patch_info(shift_start, shift_stop, scale_start, scale_stop, n)

        patches = create_patches(patches[0], list_tx_ty, scales, add_extra_patches=adding_extra_patches)
        frames = generate_frames(G_base, patches, ws_list[i], write_text_on_images=write_text_on_images)
        write_frames_to_file(frames, output_folder_and_file, output_format)



def create_patch_info_split(output_resolution=1024, grid=4, PIXEL_LEN = 1/1024, use_linspace:bool = False):
    l = []
    if use_linspace:
        SUB_PIXEL = PIXEL_LEN / 2 # DEPREICATED METHOD??
        tx_matrix = np.linspace(-2*SUB_PIXEL, 2*SUB_PIXEL, 4).reshape(1,4).repeat(4, axis=0)
        ty_matrix = np.linspace(-2*SUB_PIXEL, 2*SUB_PIXEL, 4).reshape(4,1).repeat(4, axis=1)
    else:
        limit = PIXEL_LEN - PIXEL_LEN / grid # 1 - 1/4 = 3/4, from 0, 1/4, 2/4, 3/4
        tx = torch.tensor([[0, limit], [0, limit]]).unsqueeze(0).unsqueeze(0)
        ty = torch.tensor([[0, 0], [limit, limit]]).unsqueeze(0).unsqueeze(0)
        tx_matrix = torch.nn.functional.interpolate(tx, size=4, mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
        
        ty_matrix = torch.nn.functional.interpolate(ty, size=4, mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
    
    tx_matrix = tx_matrix if isinstance(tx_matrix, torch.Tensor) else torch.from_numpy(tx_matrix)
    ty_matrix = ty_matrix if isinstance(ty_matrix, torch.Tensor) else torch.from_numpy(ty_matrix)

    for i in range(grid):
        for j in range(grid):
            matrix = torch.eye(3)
            matrix[0, 2] = tx_matrix[i, j] # tx
            matrix[1, 2] = ty_matrix[i, j] # ty
            l.append(((0, output_resolution, 0, output_resolution), matrix))
    return l, tx_matrix, ty_matrix

def combine_split_intertwine_pixels(frames:List[torch.Tensor]) -> torch.Tensor:
    assert frames is not None
    assert isinstance(frames, list) 
    
    stacked = torch.stack(frames, dim=0).permute(0, 2, 3, 1)
    assert isinstance(stacked, torch.Tensor), f"Expected torch.Tensor, got {type(frames[0])}"
    img_size = frames[0].shape[1]
    
    w = einops.rearrange(stacked, '(b1 b2) (h1 h2) (w1 w2) c -> (h1 b1 h2) (w1 b2 w2) c', h1=img_size, w1=img_size, b2=4, c=3).permute(2, 0, 1)
    if img_size == 1024:
        assert w is not None and w.shape == (3, 4096, 4096) # TODO:not quite sure if I will use more than 4x4 images and then downsample?
    return w


def generate_different_images_split(selected, ws_list, G_base, output_resolution=4096, output_format="jpg", grid=4, extra:str="p000000/", mov_scale = 1, num_pixels = 1024):
    """_summary_

    Args:
        selected (_type_): _description_
        ws_list (_type_): _description_
        G_base (_type_): _description_
        output_resolution (int, optional): _description_. Defaults to 4096.
        output_format (str, optional): _description_. Defaults to "jpg".
        grid (int, optional): _description_. Defaults to 4.
        extra (str, optional): _description_. Defaults to "p000000/".
        mov_scale (float, optional): mov_scale / num_pixels => PIXEL_LEN. Defaults to 1.
        num_pixels (int, optional): 1 / num_pixels => PIXEL_LEN. Defaults to 4096. 
    """
    PIXEL_LEN = mov_scale/(num_pixels)
    folder_location = f"outputs/{extra}g{grid}_m{mov_scale}_n{num_pixels}"
    print(f"Saving to {folder_location}")
    for i in tqdm(selected, desc="Generating different images", leave=False):
        # output includes generator steps, 
        output_folder_and_file = f"{folder_location}/result_{i}"
        patches, tx_matrix, ty_matrix = create_patch_info_split(output_resolution, grid=grid, PIXEL_LEN=PIXEL_LEN)
        frames = generate_frames(G_base, patches, ws_list[i], write_text_on_images=False)
        final_frame = combine_split_intertwine_pixels(frames)
        write_frames_to_file(final_frame, output_folder_and_file, output_format)

    with open(f"{folder_location}/matrixes.txt", "w") as f:
        f.write(f"Parameters: grid={grid}, mov_scale={mov_scale}, num_pixels={num_pixels}, PIXEL_LEN={PIXEL_LEN:8f}\n")
        f.write(f"tx_matrix\n {tx_matrix}\nty_matrix \n{ty_matrix}")

if __name__ == "__main__":
    """Example run, with appended
    ./setup40.sh && python gen_videos.py --G_base training-runs/SS750/00010-stylegan3-r-LR-gpus4-batch32-gamma2/network-snapshot-000000.pkl --ws_list ws.pkl --output_resolution 1024 --output_format mp4 --shift_start -0.5 --shift_stop 0.5 --scale_start 0.5 --scale_stop 1.5 --n 8 --extra r000000/
    
    
    module purge && module load gcc && module unload zlib && module load cudnn && module unload cuda/10.2.89 && module unload libiconv/1.16 && module unload xz && module unload zlib && module unload libxml2 && module load cuda/11.5.0 && python gen_videos.py --G_base training-runs/SS750/00010-stylegan3-r-LR-gpus4-batch32-gamma2/network-snapshot-000000.pkl --ws_list ws.pkl --output_resolution 1024 --output_format mp4 --shift_start -0.5 --shift_stop 0.5 --scale_start 0.5 --scale_stop 1.5 --n 8 --extra r000000/
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--G_base", type=str, help="Location of the G_base file")
    parser.add_argument("--ws_list", type=str, help="Location of the ws_list file")
    parser.add_argument("--output_resolution", type=int, default=1024, help="Output resolution of the images")
    parser.add_argument("--output_format", type=str, default="mp4", help="Output format of the images")
    parser.add_argument("--shift_start", type=float, default=-0.25, help="Start value for shift")
    parser.add_argument("--shift_stop", type=float, default=0.25, help="Stop value for shift")
    parser.add_argument("--scale_start", type=float, default=1, help="Start value for scale")
    parser.add_argument("--scale_stop", type=float, default=1, help="Stop value for scale")
    parser.add_argument("--n", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--extra", type=str, default="r000000/", help="Extra path for the output images")

    args = parser.parse_args()

    G_base, ws_list = load_G(args.G_base), load_ws(args.ws_list)
    generate_different_images(list(range(len(ws_list))),
                              ws_list,
                              G_base,
                              output_resolution=args.output_resolution,
                              output_format=args.output_format,
                              shift_start=args.shift_start,
                              shift_stop=args.shift_stop,
                              scale_start=args.scale_start,
                              scale_stop=args.scale_stop,
                              n=args.n, extra=args.extra)