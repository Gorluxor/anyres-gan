from tqdm import tqdm
from typing import List, Tuple
import torchvision
import os
from training.networks_stylegan3 import Generator

from ipywidgets import HTML
from ipyevents import Event 
from IPython.display import display
import torch
import pickle
from util import renormalize, viz_util, patch_util
import numpy as np
torch.set_grad_enabled(False)

# from torch_utils import persistence

# @persistence.persistent_class
class MappingNetworkZero(torch.nn.Module):
    def __init__(self, z_dim, c_dim, w_dim, num_ws):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        # Return a tensor of zeros with the correct dimensions
        return torch.zeros((z.shape[0], self.num_ws, self.w_dim)).to(z.device)

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'



def create_random_grid_1k(G_base, grid_size:int = 4, random_state:int = 19, truncation_psi:int=0.5, verbose:bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    items, ws_list = [], []
    for i in tqdm(range(grid_size*grid_size), desc='Generating images 1k', disable= not verbose):
        rng = np.random.RandomState(random_state+i)
        z = torch.from_numpy(rng.standard_normal(G_base.z_dim)).float()
        z = z[None].cuda()
        c = None
        ws = G_base.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=None) # default 0.5, 8
        img = patch_util.scale_condition_wrapper(G_base, ws, transform=None, noise_mode='const', force_fp32=True)
        items.append(renormalize.as_tensor(img[0], target='pt')) # 1 x 3 x 1024 x 1024
        ws_list.append(ws)   # 1 x 16 x 512
    # pickle it 
    # with open(f'ws.pkl', 'wb') as f:
    #     pickle.dump(ws_list, f)
    return items, ws_list

def create_random_grid_4k(G_base, ws_list:List[torch.Tensor], patches:List = None, grid_size:int = 4, random_state:int = 19, full_size:int = 4096, fs:int = None, verbose:bool = False) -> List[torch.Tensor]:
    items = []
    for i in tqdm(range(grid_size*grid_size), desc='Generating images 4k', disable=not verbose):        
        full = torch.zeros([1, 3, full_size, full_size])
        if patches is None:
            patches = patch_util.generate_full_from_patches(full_size, G_base.img_resolution)
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

def create_random_grid(G_base, grid_size, random_state:int = 19, full_size:int = 4096, verbose:bool = False, fs:int = None, only_1k:bool = False):
    imgs, ws_list = create_random_grid_1k(G_base, grid_size, random_state, truncation_psi=0.5, verbose=verbose)
    imgs_4k = create_random_grid_4k(G_base, ws_list, grid_size, random_state, full_size, fs=fs, verbose=verbose) if not only_1k else None
    return imgs, imgs_4k
import PIL
def create_grid_main(G_base:torch.nn.Module, grid_size:int = 4,
                     random_state:int = 19, full_size:int = 4096,
                     save_name:str = None, ext:str = ".jpg", 
                     folder:str = "output", verbose:str = False, fs:int = None, only_1k:bool = False):
    """Create a grid of random images. using 1k base and then creating the 4k equivalent

    Args:
        G_base (torch.nn.Module): Generator
        grid_size (int, optional): n x n. Defaults to 4.
        random_state (int, optional): Random state to start from, +1 for each of grid. Defaults to 19.
        full_size (int, optional): Final size. Defaults to 4096.
        save_name (str, optional): _description_. Defaults to None.
        ext (str, optional): _description_. Defaults to ".jpg".
        folder (str, optional): _description_. Defaults to "output".
        verbose (str, optional): _description_. Defaults to False.
        only_1k (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """    
    imgs, imgs_4k = create_random_grid(G_base, grid_size, random_state, full_size, verbose=verbose, fs=fs, only_1k=only_1k)
    grid = torchvision.utils.make_grid(imgs, nrow=grid_size) 
    grid_4k = torchvision.utils.make_grid(imgs_4k, nrow=grid_size) if not only_1k else None
    if save_name is not None:
        if not os.path.exists(folder):
            os.makedirs(folder)
        base_name = f"{folder}/grid_{save_name}{ext}"
        hr_name = f"{folder}/grid_{save_name}_4k{ext}"
        verbose_text = f"Saving {base_name} and {hr_name}..." if not only_1k else f"Saving {base_name}..."
        if verbose: print(verbose_text)
        img = (grid.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8) # 1 x 3 x 1024 x 1024 -> 1 x 1024 x 1024 x 3, without batch size it would be .permute(1, 2, 0)
        #PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(f'{base_name}')
        torchvision.utils.save_image(grid, base_name)
        if not only_1k:
            img_4k = (grid_4k.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            #PIL.Image.fromarray(img_4k.cpu().numpy(), 'RGB').save(f'{hr_name}')
            torchvision.utils.save_image(grid_4k, hr_name)
        return 
    return grid, grid_4k

def load_and_grid(G_path, grid_size:int = 4, random_state:int = 19,
                  full_size:int = 4096, save_name:str = None,
                  ext:str = ".jpg", folder:str = "output",
                  verbose:str = False, fs:int = None, force_MN:bool = False, only_1k:bool = False):
    import re
    # at least 6 digits
    # re template: r'(\d{6})'
    
    #current_idx =  #G_path.split('-')[-1].split('.')[0] # XXXXXX number
    if save_name is None:
        template = re.compile(r'(\d{6})')
        current_idx = template.findall(G_path)[-1] if len(template.findall(G_path)) > 0 else "base"
        save_name = current_idx
    with open(G_path, 'rb') as f:
        G_base: Generator = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
    if force_MN:
        print('a')
        # get z_dim, c_dim, w_dim, num_ws from G_base.scale_mapping and replace it with MappingNetworkZero
        print(G_base.scale_mapping)
        G_base.scale_mapping = MappingNetworkZero(G_base.scale_mapping.z_dim, G_base.scale_mapping.c_dim, G_base.scale_mapping.w_dim, G_base.scale_mapping.num_ws).cuda()
        print(G_base.scale_mapping)
        # G_base.scale_mapping_kwargs = None
        # G_base.use_scale_affine = None
        # #{
        # #     'scale_mapping_min': 1,
        # #     'scale_mapping_max': 1,
        # #     'scale_mapping_norm': 'positive',
        # # }
    return create_grid_main(G_base, grid_size, random_state, full_size, save_name, ext, folder, fs=fs, verbose=verbose, only_1k=only_1k)


def revert_weights(model_teacher, model_new, verbose=False):
    for name, param in model_new.named_parameters():
        value = model_teacher.state_dict().get(name, None)
        if value is not None:
            model_new.state_dict()[name].copy_(value)
        else:
            if verbose:
                print(name, param.shape, "not found")

def compare_models(model_teacher, model_new, verbose=False):
    total_diff = 0
    for name, param in model_new.named_parameters():
        value = model_teacher.state_dict().get(name, None)
        if value is not None:
            diff = torch.sum((param - value)**2)
            total_diff += diff
            if verbose:
                print(name, param.shape, value.shape, f"l2{diff.item()=}")
        else:
            if verbose:
                print(name, param.shape, "not found")
    final_str = f"total_diff={total_diff}"
    if verbose:
        print(final_str)
    return final_str


if __name__ == "__main__":
    from argparse import ArgumentParser
    """
    python grids.py --G_path training-runs/SS750/00010-stylegan3-r-LR-gpus4-batch32-gamma2/network-snapshot-000000.pkl --grid_size 4 --random_state 19 --full_size 4096 --save_name 000000 --ext .jpg --folder output --verbose True --force_MN False
    """
    parser = ArgumentParser()
    parser.add_argument('--G_path', type=str, required=True)
    parser.add_argument('--grid_size', type=int, default=4)
    parser.add_argument('--random_state', type=int, default=19)
    parser.add_argument('--full_size', type=int, default=4096)
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--ext', type=str, default=".jpg")
    parser.add_argument('--folder', type=str, default="output")
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--only_1k', type=bool, default=False)
    parser.add_argument('--force_MN', type=bool, default=False)
    parser.add_argument('--fs', type=int, default=None)
    args = parser.parse_args()
    load_and_grid(**vars(args))