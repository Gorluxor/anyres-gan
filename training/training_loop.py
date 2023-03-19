# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main

from util import util
from util import patch_util
import random
from metrics import equivariance
from torchvision.transforms import functional as F_tv
from torchvision.transforms import InterpolationMode
#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size, q:int = 85):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    # check if fname has .jpg extension and set quality to 85 
    extra_args = dict(quality=q) if fname.endswith('.jpg') else {}
    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname, **extra_args)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname, **extra_args)

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for base training set.
    patch_kwargs     = {},         # Options for patch dataset.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    added_kwargs = {}, # added
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    # ADDED: to prevent data_loader pin_memory to load to device 0 for every process
    torch.cuda.set_device(device)
    training_mode = G_kwargs.training_mode

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))

    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Load patch dataset
    if 'patch' in training_mode:
        if rank == 0:
            print('Loading patch dataset...')
        patch_dset = dnnlib.util.construct_class_by_name(**patch_kwargs) # subclass of training.dataset.Dataset
        # with open('testa/patch_kwargs.pkl', 'wb') as f:
        #     pickle.dump(patch_kwargs, f)
        patch_dset_sampler = misc.InfiniteSampler(dataset=patch_dset, rank=rank, num_replicas=num_gpus, seed=random_seed)
        patch_dset_iterator = iter(torch.utils.data.DataLoader(dataset=patch_dset, sampler=patch_dset_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
        if rank == 0:
            print()
            print('Patch Num images: ', len(patch_dset))
            print('Patch Image shape:', patch_dset.image_shape)
            print('Patch Label shape:', patch_dset.label_shape)
            print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')

    # modified: use specified img_resolution
    img_resolution = training_set.resolution
    if 'patch' in training_mode and added_kwargs.img_size is not None:
        img_resolution = added_kwargs.img_size
        if rank == 0:
            print("Using specified img resolution: %d" % img_resolution)
        assert(added_kwargs.img_size == training_set.resolution)
    num_cdim = training_set.label_dim + int(added_kwargs.bcondg)
    num_cdim_d = training_set.label_dim + 4 * int(added_kwargs.bcond) + int(added_kwargs.bcondextra)
    common_kwargs = dict(c_dim=num_cdim, img_resolution=img_resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    if added_kwargs.overrided: # change D img_resolution to be 1/4
        common_kwargs['img_resolution'] = img_resolution // 4
        D_kwargs['channel_base'] = D_kwargs['channel_base'] // 2
        assert added_kwargs.use_hr == True, "Use hr must be true when overrided"
    common_kwargs['c_dim'] = num_cdim_d
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    if 'patch' in training_mode and added_kwargs.teacher is not None:
        teacher = copy.deepcopy(G).to(device).eval()
    if added_kwargs.use_hr and added_kwargs.use_teached_layers:
        reset_value_dict = dnnlib.EasyDict()
        for layer_name in added_kwargs.use_teached_layers:
            reset_value_dict[layer_name] = teacher.state_dict()[layer_name]

    if added_kwargs.use_hr == True and 'patch' in training_mode:
        common_kwargs_4k = dict(c_dim=num_cdim, actual_resolution=img_resolution, img_resolution=added_kwargs.actual_resolution, img_channels=training_set.num_channels)
        if rank == 0:
            print('Changing network to 4k version')
        G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs_4k).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        
    G_ema = copy.deepcopy(G).eval()    
    if added_kwargs.use_hr:
        G.synthesis.add_reset_layers(reset_value_dict)
        G_ema.synthesis.add_reset_layers(reset_value_dict)
    # copy G for teacher network: copy teacher G_ema to G_ema:,
    # uses G state dict for the generator to align with D
    if 'patch' in training_mode and added_kwargs.teacher is not None:
        # deactivate scale affine adding in teacher model; so it matches original model
        for layer_name in teacher.synthesis.layer_names:
            layer = getattr(teacher.synthesis, layer_name)
            layer.use_scale_affine = False

        if rank == 0:
            print(f"loading teacher from {added_kwargs.teacher} on device %s! " % rank)
            with dnnlib.util.open_url(added_kwargs.teacher) as f:
                teacher_data = legacy.load_network_pkl(f)
            # with open('testa/NOW_teacher.txt', 'w') as f:
            #     print(f'Teacher: {teacher_data["G"]}', file=f)
            #     print(f'Teacher D: {teacher_data["D"]}', file=f)
            # with open('testa/NOW_actual.txt', 'w') as f:
            #     print(f'Actual: {G}', file=f)
            #     print(f'Actual D: {D}', file=f)
            for name, module in [('G', G), ('G_ema', teacher), ('G_ema', G_ema), ('D', D)]:
                print('Copying', name)
                if added_kwargs.reinitd and name == 'D': # skip copying discriminator
                    continue
                if added_kwargs.overrided and name == 'D': # load from a different discriminator
                    with dnnlib.util.open_url(added_kwargs.overrided) as f:
                        misc.copy_params_and_buffers(legacy.load_network_pkl(f)['D'], module, require_all=False, allow_ignore_different_shapes=False)
                    continue
                misc.copy_params_and_buffers(teacher_data[name], module, require_all=False, allow_ignore_different_shapes=added_kwargs.use_hr)
            print(f"done loading teacher on device %s! " % rank)
            # util.set_requires_grad(False, teacher)
    else:
        teacher = None

    if teacher is not None:
        teacher.synthesis.remove_all_delta_weights(rank)
    G.reconfigure_network(img_resolution=added_kwargs.actual_resolution, use_old_filters=added_kwargs.use_old_filters)
    G_ema.reconfigure_network(img_resolution=added_kwargs.actual_resolution, use_old_filters=added_kwargs.use_old_filters)
    already_done = False # just for teacher images
    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False, allow_ignore_different_shapes=added_kwargs.use_hr)
    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        print('--- Teacher ---')
        img = misc.print_module_summary(teacher, [z, c])
        print('--- Discriminator ---')
        if added_kwargs.overrided:
            # Downsample images, just for printing
            from torch.nn import functional as F
            img = F.interpolate(img, size=(img_resolution // 4, img_resolution // 4), mode='bilinear', align_corners=False)
        cd = torch.empty([batch_gpu, D.c_dim], device=device)
        misc.print_module_summary(D, [img, cd])
        del img
        torch.cuda.empty_cache()
        print('--- Generator ---')
        img = misc.print_module_summary(G, [z[:1,:], c[:1,:]])
        del z, c, img
        torch.cuda.empty_cache()
    
    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for name, module in [('G', G), ('D', D), ('G_ema', G_ema),
                         ('teacher', teacher), ('augment', augment_pipe)]:
        if module is not None and num_gpus > 1:
            if rank == 0:
                print("copied %s across gpus!" % name)
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)
        elif module is None:
            if rank == 0:
                print("%s is None; not copied!" % name)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe,
                                               added_kwargs=added_kwargs, teacher=teacher, **loss_kwargs) # subclass of training.loss.Loss

    phases = []
    # return G, G_ema, loss
    for name, module, params, opt_kwargs, reg_interval in [('G', G, G.parameters(), G_opt_kwargs, G_reg_interval),
                                                           ('D', D, D.parameters(), D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=params, **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(params, **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)
    if rank == 0:
        print(f'Existing phases: {[phase.name for phase in phases]}')
    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    # empty cuda
    torch.cuda.empty_cache()
    if rank == 0:
        with torch.no_grad():
            print('Exporting sample images...')
            grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
            save_image_grid(images, os.path.join(run_dir, 'reals.jpg'), drange=[0,255], grid_size=grid_size)
            grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
            if not added_kwargs.bcondg:
                grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
                grid_c_extra = None
            else: # add to grid_c, 0's, for old domain => append to each grid_c
                grid_c = torch.from_numpy(labels).to(device)
                grid_c_extra = torch.cat((grid_c, torch.ones(grid_c.shape[0], 1).to(device)), dim=1).split(batch_gpu)
                grid_c = torch.cat((grid_c, torch.zeros(grid_c.shape[0], 1).to(device)), dim=1).split(batch_gpu)

            del images
            if added_kwargs.use_hr:
                slice_ranges_4k = patch_util.generate_full_from_patches_slices(added_kwargs.actual_resolution, G_ema.actual_resolution, device=device)
                images = patch_util.reconstruct_image_from_patches(torch.stack([torch.cat([G_ema(z=z, c=c, noise_mode='const', slice_range=sl.repeat(z.shape[0], 1)).cpu() for z, c in zip(grid_z, grid_c)]) for sl in slice_ranges_4k]))
            else:
                images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            save_image_grid(images, os.path.join(run_dir, 'fakes_init.jpg'), drange=[-1,1], grid_size=grid_size)
            print('Done exporting sample images...')
            del images

    torch.cuda.empty_cache()
    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)
    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    if rank == 0:
        print(f"Model:{str(G)}")
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    import torchvision
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:
        with torch.autograd.profiler.record_function('data_fetch'):
            if 'patch' in training_mode:
                if random.uniform(0, 1) > added_kwargs.base_probability:
                    # base dataset iterator
                    phase_real_img, phase_real_c = next(training_set_iterator)
                    n = phase_real_img.shape[0]
                    phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
                    phase_real_c = phase_real_c.to(device).split(batch_gpu)
                    transform = torch.eye(3)[None]
                    phase_transform = transform.repeat(n, 1, 1).to(device).split(batch_gpu)
                    min_scale = 1.0
                    max_scale = 1.0
                    # if use_hr, we are going to generate random split_range, and coresponding coordinates
                    # this will be later used to crop the original image?
                    # produce 
                    if added_kwargs.use_hr:
                        # split_range_values = patch_util.generate_random_positions_tensor(n)
                        # coords = patch_util.grid2pixel_tensor_f(split_range_values.clone(), 256, 64).split(batch_gpu)
                        # split_range = split_range_values.split(batch_gpu)
                        # split_range 
                        split_range = torch.tensor([0, 36, 0, 36]).repeat(batch_size).split(batch_gpu) # identity
                        coords = [None] * len(phase_real_c)
                        print("WARNING: this is still not tested...")
                        raise NotImplementedError("Even though probably will work, logically not tested")
                        # TODO: maybe parametrize this better? now it only works for x4
                        # In this case, there are coords to be cropped on the 1k teacher supervision
                    else:
                        split_range = [None] * len(phase_real_c)
                        coords = [None] * len(phase_real_c)
                else:
                    # patch dataset iterator
                    data, phase_real_c = next(patch_dset_iterator)
                    phase_real_img = (data['image'].to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
                    phase_real_c = phase_real_c.to(device).split(batch_gpu)
                    if added_kwargs.log_imgs:
                        for iw, curr_img in enumerate(phase_real_img):
                            if not os.path.exists(f'patches_f/{iw}_{cur_tick}.jpg'):
                                torchvision.utils.save_image(curr_img, f'patches_f/{iw}_{cur_tick}.jpg', range=(-1, 1), normalize=True, nrow=4)
                    phase_transform = data['params']['transform'].to(device).split(batch_gpu)
                    if added_kwargs.use_hr:
                        split_range_data = torch.stack(data['params']['split_range']).permute(1,0).to(device)
                        # those are 1k crops coords for the teacher, not for the actual 
                        coords = patch_util.grid2pixel_tensor_f(split_range_data.clone(), 256, 64).split(batch_gpu) #TODO: parameterize this?
                        split_range = split_range_data.split(batch_gpu)
                    else:
                        split_range = [None] * len(phase_real_c)
                        coords = [None] * len(phase_real_c)
                    min_scale = data['params']['min_scale_anneal'][0].item()
                    max_scale = 1.0
            else:
                phase_real_img, phase_real_c = next(training_set_iterator)
                phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
                phase_real_c = phase_real_c.to(device).split(batch_gpu)
                # dummy variables
                phase_transform = [None] * len(phase_real_c)
                coords = [None] * len(phase_real_c)
                split_range = [None] * len(phase_real_c)
                min_scale = 1.0
                max_scale = 1.0

            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]
                        
        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            if phase.name == 'none':
                raise ValueError(f"Phase is None, this should not happen, {phase}, {phase.name}, {phase.interval}")
            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            for transform, real_img, real_c, gen_z, gen_c, curr_split, curr_coords in zip(phase_transform, phase_real_img, phase_real_c, phase_gen_z, phase_gen_c, split_range, coords):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, transform=transform,
                                          gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg,
                                          min_scale=min_scale, max_scale=max_scale, split=curr_split, coords=curr_coords)
            phase.module.requires_grad_(False)
            if added_kwargs.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(phase.module.parameters(), 1.0)
            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        
        fields += [f'G/loss {stats_collector["Loss/G/loss"]:<3.2f}']
        teacher_loss = stats_collector["Loss/G/loss_teacher_l1"] + stats_collector["Loss/G/loss_teacher_lpips"]
        fields += [f'G/loss_teacher {teacher_loss:<3.2f}']
        fields += [f'D/loss {stats_collector["Loss/D/loss"]:<3.2f}']
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        torch.cuda.empty_cache()
        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            if not added_kwargs.use_hr or added_kwargs.log_HR:
                images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
                save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.jpg'), drange=[-1,1], grid_size=grid_size)
            if added_kwargs.use_hr: #and added_kwargs.img_size != added_kwargs.actual_resolution:
                # save lower res images
                torch.cuda.empty_cache()
                low_res, high_res = added_kwargs.img_size, added_kwargs.actual_resolution
                if added_kwargs.log_LR:
                    G_ema.reconfigure_network(img_resolution=low_res, use_old_filters=added_kwargs.use_old_filters)
                    images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
                    save_image_grid(images, os.path.join(run_dir, f'fakes_lr_{cur_nimg//1000:06d}.jpg'), drange=[-1,1], grid_size=grid_size)
                    G_ema.reconfigure_network(img_resolution=high_res, use_old_filters=added_kwargs.use_old_filters)
                    torch.cuda.empty_cache()
                # high_res should be 4096 or 1024, whereas low_res is going to be 1024 and 1024
                slice_ranges_4k = patch_util.generate_full_from_patches_slices(high_res, low_res, device=device)
                images = patch_util.reconstruct_image_from_patches(torch.stack([torch.cat([G_ema(z=z, c=c, noise_mode='const', slice_range=sl.repeat(z.shape[0], 1)).cpu() for z, c in zip(grid_z, grid_c)]) for sl in slice_ranges_4k]))
                save_image_grid(images, os.path.join(run_dir, f'fakes_hr_{cur_nimg//1000:06d}.jpg'), drange=[-1,1], grid_size=grid_size)
                if grid_c_extra:
                    images = patch_util.reconstruct_image_from_patches(torch.stack([torch.cat([G_ema(z=z, c=c, noise_mode='const', slice_range=sl.repeat(z.shape[0], 1)).cpu() for z, c in zip(grid_z, grid_c_extra)]) for sl in slice_ranges_4k]))
                    save_image_grid(images, os.path.join(run_dir, f'fakes_hr_extra_{cur_nimg//1000:06d}.jpg'), drange=[-1,1], grid_size=grid_size)
                # functional interpolate with bilinear
                images = F_tv.resize(images, (low_res, low_res), interpolation=InterpolationMode.BILINEAR)
                save_image_grid(images, os.path.join(run_dir, f'fakes_hrds_{cur_nimg//1000:06d}.jpg'), drange=[-1,1], grid_size=grid_size)    

            # also save 1k images with teacher model
            if added_kwargs.teacher is not None and not already_done:
                images = torch.cat([teacher(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
                save_image_grid(images, os.path.join(run_dir, f'fakes_teacher.jpg'), drange=[-1,1], grid_size=grid_size)
                already_done = True
            del images
            torch.cuda.empty_cache()

        # if num_gpus > 1:
        #     torch.distributed.barrier()
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(G=G, D=D, G_ema=G_ema, augment_pipe=augment_pipe, training_set_kwargs=dict(training_set_kwargs))
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    if num_gpus > 1:
                        misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        for param in misc.params_and_buffers(value):
                            torch.distributed.broadcast(param, src=0)
                    snapshot_data[key] = value.cpu()
                del value # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                    dataset_kwargs=training_set_kwargs, patch_kwargs=patch_kwargs, num_gpus=num_gpus, rank=rank, device=device, extra={"bcondg": added_kwargs.bcondg})
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
