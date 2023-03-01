import numpy as np
import random
import torch
import math
from math import exp
from PIL import Image
import random


def construct_transformation_matrix(limits):
    # limits is a list of [(y_min, y_max), (x_min, x_max)]
    # in normalized coordinates from -1 to 1
    x_limits = limits[1]
    y_limits = limits[0]
    theta = torch.zeros(2, 3)
    tx = np.sum(x_limits) / 2
    ty = np.sum(y_limits) / 2
    s = x_limits[1] - tx
    assert(np.abs((x_limits[1] - tx) - (y_limits[1] - ty)) < 1e-9)
    theta[0, 0] = s
    theta[1, 1] = s
    theta[0, 2] = (tx) / 2
    theta[1, 2] = (ty) / 2
    transform = torch.zeros(3, 3)
    transform[:2, :] = theta
    transform[2, 2] = 1.0
    return transform

class PatchSampler(object):
    def __init__(self, patch_size, random_shift=True, random_scale=True, scale_anneal=-1,
                 max_scale=None, min_scale=None, use_hr=None, latent_len:int = 24, use_normal:bool = False, latent_spacing = 4, **kwargs):
        self.patch_size = patch_size # variable p in paper
        self.random_shift = random_shift
        self.random_scale = random_scale
        # full image range is [-1, 1]
        self.w = np.array([-1, 1])
        self.h = np.array([-1, 1])
        # image size = 1/scale
        self.max_scale = max_scale if max_scale is not None else 1.0
        self.min_scale = min_scale
        self.iterations = 0
        self.scale_anneal = scale_anneal
        self.initial_min = 1.0 # 0.9
        self.use_hr = use_hr # ADDED
        self.latent_len = latent_len # TODO:My: parametrize this outside PatchSampler
        self.use_normal = use_normal

    def sample_patch(self, im):
        im_w, im_h  = im.size
        assert(im_w == im_h) # crop to square image before patch sampling

        # minimum scale bound based on image size
        min_scale = self.patch_size / im_h
        if self.min_scale is not None:
            min_scale = max(min_scale, self.min_scale)
        params = {'min_scale_absolute': min_scale}

        #  adjust min scale if annealing
        if self.scale_anneal > 0:
            k_iter = (self.iterations)// 1000 * 3
            # decays min_scale between self.min_scale and initial_min
            min_scale = max(min_scale, self.max_scale * exp(-k_iter*self.scale_anneal))
            min_scale = min(self.initial_min, min_scale)
        params['min_scale_anneal'] = min_scale

        scale = 1.0
        if self.random_scale:
            # this samples is size uniformly from min_size to max_size
            max_size = self.patch_size / min_scale
            min_size = self.patch_size / self.max_scale
            random_size = random.uniform(min_size, max_size)
            scale = self.patch_size / random_size
        params['sampled_scale'] = scale

        # resize the image to a random new size and take a crop
        new_size = int(np.round(self.patch_size / scale))
        crop_size = self.patch_size
        if not self.use_hr:
            im_resized = im.resize((new_size, new_size), Image.LANCZOS)
            assert(new_size <= im_h) # do not upsample
            x = random.randint(0, np.maximum(0, new_size - crop_size)) # inclusive [low, high]
            y = random.randint(0, np.maximum(0, new_size - crop_size))
        else: # no resizing, anyway, fixed size, descrite grid size
            # TODO: My: maybe consolidate it in 1 function?
            im_resized = im
            if self.use_normal: # use normal distribution on the int distribution from [0, 12] instead of uniform
                x = max(0, min(12, round(random.normalvariate(6, 2))))
            else:
                x = random.randint(0, 12)
            y = random.randint(0, 12)
            params['split_range'] = (x, x+self.latent_len, y, y+self.latent_len)
            actual_coords = grid2pixel(params['split_range'], r=True)
            params['coords'] = actual_coords[0] if isinstance(actual_coords, list) else actual_coords # returns a list # TODO: maybe param this?
            
            x = params['coords'][0] # calculate actual coords (x_start, x_end, y_start, y_end)
            y = params['coords'][2] # calculate actual coords

        im_crop = im_resized.crop((x, y, x+crop_size, y+crop_size))

        # normalized limits
        limits = [(y/(new_size)*2-1, (y+crop_size) /(new_size)*2-1),
                  (x/(new_size)*2-1, (x+crop_size) /(new_size)*2-1)]
        params['limits'] = limits
        params['x'] = x
        params['y'] = y
        params['new_size'] = new_size
        params['orig_size'] = im_w

        # calculate the transformation matrix
        if not self.use_hr:
            transform = construct_transformation_matrix(limits)
        else:
            transform = torch.eye(3,3) # no tx and ty scaling
        # make a tensor [(x, y, x+crop_size, y+crop_size)] with the same structure as [x, y, x+crop_size, y+crop_size]
        #crop_params = torch.tensor([x, y, x+crop_size, y+crop_size]) #.unsqueeze(0)
        #params['crop_params'] = torch.tensor([x, y, x+crop_size, y+crop_size], dtype=torch.long, requires_grad=False)
        params['transform'] = transform

        return im_crop, params

def generate_full_from_patches(new_size, patch_size=256):
    # returns the bounding boxes and transformations needed to 
    # piece together patches of size patch_size into a 
    # full image of size new_size
    patch_params = []
    for y in range(0, new_size, patch_size):
        for x in range(0, new_size, patch_size):
            if y + patch_size > new_size:
                y = new_size - patch_size
            if x + patch_size > new_size:
                x = new_size - patch_size
            limits = [(y/(new_size)*2-1, (y+patch_size) /(new_size)*2-1),
              (x/(new_size)*2-1, (x+patch_size) /(new_size)*2-1)]
            transform = construct_transformation_matrix(limits)
            patch_params.append(((y, y+patch_size, x, x+patch_size), transform))
    return patch_params

def transform_back_to_slice(coords, img_size=1024, step_len=24, device:torch.device=None):
    """ Transform from px slice range i.e. (0, 1024, 0, 1024) to latent space

    Args:
        coords (_type_): _description_
        img_size (int, optional): _description_. Defaults to 1024.
        step_len (int, optional): _description_. Defaults to 24.
        device (torch.device, optional): If given, return tensor. Defaults to None.

    Raises:
        NotImplementedError: TODO: add other resolutions

    Returns:
        Torch.tensor | List[Tuple]: Depends on device
    """    
    if step_len != 24:
        raise NotImplementedError('Implement other than 4096 and 1k')
    if isinstance(coords, tuple):
        coords = [coords]
    step = img_size // 4
    rez = [(x // step, x//step + step_len, y // step, y // step + step_len) for x, x_end, y, y_end in coords]
    if device is not None:
        rez = torch.tensor(rez, device=device)
    return rez

def get_list_of_slices(max_len=24, skip=4):
    """Generate list of transforms to produce a 4k image, from 1k segments
    Works for other resolutions, just need to change max_items, for 1k it is 28 and 

    Args:
        max_len (int, optional): _description_. Defaults to 24.
        skip (int, optional): How many features to skip for the next box. Defaults to 4.
    """
    # potential_list = [((k, k + max_len, k, k + max_len)) for k in range(0, 36, skip)]
    potential_list = []
    for i in range(0, 36, skip):
        for k in range(0, 36, skip):
            if k + max_len <= 36 and i + max_len <= 36:
                potential_list.append((k, k + max_len, i, i + max_len))
    
    return potential_list

def get_grid_params(resolution:int = 4096):
    """Get the grid parameters for a given resolution

    Args:
        resolution (int, optional): Resolution of the image. Defaults to 4096.

    Returns:
        list: List of tuples of the form (x, y)
    """
    if resolution == 4096:
        return get_list_of_slices(24, 4)
    elif resolution == 1024:
        return NotImplementedError("Theoreticall need to figure out") # get_list_of_slices(28, 3)
    else:
        raise ValueError("Resolution not supported")
    
def generate_full_from_patches_slices(new_size:int, device:torch.device=None):
    rez = get_grid_params(new_size)
    if device is not None:
        rez = torch.tensor(rez, device=device)
    return rez

import einops
def reconstruct_image_from_patches(img:torch.Tensor, hs=4, ws=4) -> torch.Tensor:
    """Reverse of einops.rearrange(img, '... c (hs h) (ws w) -> ... c (hs ws) h w  ', hs=4, ws=4)

    Args:
        img (torch.Tensor): tensor of shape (batch, n, channels, height, width) or (n, channels, height, width)
        hs (int, optional): grid split height. Defaults to 4.
        ws (int, optional): grid split width. Defaults to 4.

    Returns:
        torch.Tensor: (B, C, H*hs, W*ws)
    """    
    # batch_second = hs * ws != img.shape[-3]
    # if batch_second:
    return einops.rearrange(img, '(ws hs) b c h w -> b c (hs h) (ws w) ', hs=hs, ws=ws)
    # return einops.rearrange(img, 'b n c (hs ws) h w -> b c (hs h) (ws w)', hs=hs, ws=ws)

def compute_scale_inputs(G, w, transform):
    if transform is None:
        scale = torch.ones(w.shape[0], 1).to(w.device)
    else:
        scale = 1/transform[:, [0], 0]
    scale = G.scale_norm(scale)
    mapped_scale = G.scale_mapping(scale, None)
    return scale, mapped_scale

def scale_condition_wrapper(G, w, transform, slice_range = None, **kwargs):
    # convert transformation matrix into scale input
    # and pass through scale mapping network
    if not G.scale_mapping_kwargs or G.scale_mapping_kwargs.scale_mapping_min == G.scale_mapping_kwargs.scale_mapping_min:
        img = G.synthesis(w, transform=transform, slice_range=slice_range, **kwargs)
        return img
    scale, mapped_scale = compute_scale_inputs(G, w, transform)

    img = G.synthesis(w, mapped_scale=mapped_scale, transform=transform, slice_range=slice_range, **kwargs)
    return img

def grid2pixel_old(grid_params, res=1024, increment = 256):
    assert grid_params is not None
    assert len(grid_params) > 0
    if isinstance(grid_params, tuple):
        grid_params = [grid_params]
    assert len(grid_params[0]) == 2 # (x_start, x_end), (y_start, y_end) for slicing
    return [(tuple([k[0] * increment, k[0] * increment + res]), tuple([v[0] * increment, v[0] * increment + res])) for k, v in grid_params]


def grid2pixel(grid_params, res=1024, increment = 256, r:bool = True):
    assert grid_params is not None
    assert len(grid_params) > 0
    if isinstance(grid_params, tuple):
        grid_params = [grid_params]
    # NOTE: here, we actually change from xs, xs_end, ys, ys_end to xs, ys, xs_end, ys_end, to match the order of the grid_params for cropping
    # for x_start, x_end, y_start, y_end in grid_params, trasform to x_start, y_start, x_end, y_end
    if r:
        return [(tuple([xs * increment, xs * increment + res, ys * increment, ys * increment + res])) for xs, xend, ys, yend in grid_params]
    return [(tuple([xs * increment, ys * increment, xs * increment + res,  ys * increment + res])) for xs, xend, ys, yend in grid_params]
    #return [(tuple([k[0] * increment, v[0] * increment, k[0] * increment + res, v[0] * increment + res])) for (k), (v in grid_params]
    

def grid2pixel_tensor(grid_params, res=1024, increment=256):
    assert grid_params is not None
    assert isinstance(grid_params, torch.Tensor), f'{grid_params}'
    # clone to avoid modifying the original
    # grid_params = grid_params.clone()
    if len(grid_params.shape) == 1:
        grid_params = grid_params.unsqueeze(0) # B x 4
    
    # NOTE: here, we actually change from xs, xs_end, ys, ys_end to xs, ys, xs_end, ys_end, to match the order of the grid_params for cropping
    for i in range(grid_params.shape[0]):
        grid_params[i, 3] = grid_params[i, 2] * increment + res # lower
        grid_params[i, 1] = grid_params[i, 2] * increment # upper
        grid_params[i, 2] = grid_params[i, 0] * increment + res # right
        grid_params[i, 0] = grid_params[i, 0] * increment # left

    return grid_params

def grid2pixel_tensor_f(grid_params, res=1024, increment=256):
    assert grid_params is not None
    assert isinstance(grid_params, torch.Tensor), f'{grid_params}'
    # clone to avoid modifying the original
    # grid_params = grid_params.clone()
    if len(grid_params.shape) == 1:
        grid_params = grid_params.unsqueeze(0) # B x 4
    
    # NOTE: here, we actually change from xs, xs_end, ys, ys_end to xs, ys, xs_end, ys_end, to match the order of the grid_params for cropping
    for i in range(grid_params.shape[0]):
        grid_params[i, 3] = grid_params[i, 2] * increment + res # lower  # ys_end 
        grid_params[i, 2] = grid_params[i, 2] * increment # upper # ys
        grid_params[i, 1] = grid_params[i, 0] * increment + res # right # xs_end
        grid_params[i, 0] = grid_params[i, 0] * increment # left # xs

    return grid_params

def pil_crop_on_tensors(imgs, crops, r:bool = False):
    assert isinstance(imgs, torch.Tensor), f"{type(imgs)}, {imgs=}"
    assert isinstance(crops, torch.Tensor), f"{type(crops)}, {crops=}" #TODO: check if this is needed
    # if crops is None: # NOTE: assuming that this is the 1k patch mode
    #     return imgs
    assert len(imgs.shape) == 4
    assert len(crops.shape) == 2
    # same size of crops
    if len(crops.shape) == 1:
        crops = crops.unsqueeze(0).repeat(imgs.shape[0], 1) # B x 4 in case of 1 crop
    assert imgs.shape[0] == crops.shape[0]
    if r: # if to use xs, x_end, ys, y_end
        return torch.stack([img[:, crop[0]:crop[1], crop[2]:crop[3]] for img, crop in zip(imgs, crops)])
    return torch.stack([img[:, crop[1]:crop[3], crop[0]:crop[2]] for img, crop in zip(imgs, crops)])

def generate_rng_positions(num_positions:int, max_grid = 12, latent_size=24):
    """Generate random positions for the teacher distilation supervision, in grid space

    Args:
        num_positions (int): _description_
        max_grid (int, optional): _description_. Defaults to 12.
        latent_size (int, optional): _description_. Defaults to 24.

    Returns:
        _type_: _description_
    """
    res = lambda x,y, v: (x,x+v, y, y+v)
    return [res(random.randint(0, max_grid), random.randint(0, max_grid), latent_size)  for _ in range(num_positions)]
    
    

def generate_random_positions_tensor(num_positions:int, max_grid = 12, latent_size=24, device:torch.device=None):
    """Generate random positions for the teacher distilation supervision, in grid space

    Args:
        num_positions (int): _description_
        max_grid (int, optional): _description_. Defaults to 12.
        latent_size (int, optional): _description_. Defaults to 24.
        device (torch.device, optional): Which device should the tensors use. Defaults to None ('cpu').

    Returns:
        _type_: _description_
    """    
    
    
    if device is None:
        device = torch.device('cpu')
    x = torch.randint(0, max_grid, (num_positions, ), device=device)
    y = torch.randint(0, max_grid, (num_positions, ), device=device)
    # return as list of (x_start, x_end, y_start, y_end)
    # return list(zip(x, x+latent_size, y, y+latent_size))
    return torch.stack((x, x+latent_size, y, y+latent_size), dim=1)



