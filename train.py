# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Alias-Free Generative Adversarial Networks"."""

import os
import click
import re
import json
import tempfile
import torch

import dnnlib
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops
#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'
    # with open("testa/settings.pkl", 'wb') as f:
    #     import pickle
    #     pickle.dump(c, f)
    #     import sys
    #     sys.exit(0)
    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    if 'patch' in c.G_kwargs.training_mode:
        print(f'Patches path:        {c.patch_kwargs.path}')
        print(f'Patches size:        {c.patch_kwargs.max_size} images')
        print(f'Patches resolution:  {c.patch_kwargs.resolution}')
        print(f'Patches labels:      {c.patch_kwargs.use_labels}')
        print(f'Patches x-flips:     {c.patch_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        try:
            dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        except AssertionError:
            print("Cannot determine default dataset resolution, will try to use specified arguments")
            dataset_kwargs.resolution = None
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--cfg',          help='Base configuration',                                      type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), required=True)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed']), default='ada', show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)

# Misc hyperparameters.
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1))
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

# additional base options
@click.option('--training_mode', help='generator training mode', type=click.Choice(['global', 'patch', 'global-360']), required=True)
@click.option('--data_resolution', help='LR dataset resolution (specify if images are not preprocessed to same size and square)', type=click.IntRange(min=0))
@click.option('--random_crop', help='random crop image on LR dataset (specify if images are not preprocessed to same size and square)', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--data_max_size', help='LR dataset max number of images', type=click.IntRange(min=0))
@click.option('--g_size', help='size of G (if different from dataset size)', type=click.IntRange(min=0))
@click.option('--patch_size', help='size of patch (if different from dataset size)', type=click.IntRange(min=0), default=0)

# additional options for patch model
@click.option('--teacher', help='teacher checkpoint', metavar='[PATH|URL]',  type=str)
@click.option('--teacher_lambda', help='teacher regularization weight', metavar='FLOAT', type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--teacher_mode',  help='teacher loss mode', type=click.Choice(['inverse', 'forward', 'crop']), default='forward', show_default=True)
@click.option('--scale_anneal', help='scale annealing rate (-1 for no annealing)', metavar='FLOAT', type=click.FloatRange(min=-1), default=-1, show_default=True)
@click.option('--scale_min',  help='minimum sampled scale (leave blank to use image native resolution)', metavar='FLOAT', type=click.FloatRange(min=0))
@click.option('--scale_max',  help='maximum sampled scale', metavar='FLOAT', type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--base_probability', help='probability to sample from LR dataset with identity transform', metavar='FLOAT', type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--data_hr', help='HR patch dataset path', metavar='[ZIP|DIR]', type=str)
@click.option('--patch_crop', help='perform random cropping on non-square images (on patch dataset)', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--data_hr_max_size', help='patch dataset max number of images', type=click.IntRange(min=0))
@click.option('--scale_mapping_min', help='normalization minimum for scale mapping branch (size = g_size*scale_mapping_min)', type=click.IntRange(min=0))
@click.option('--scale_mapping_max', help='normalization maximum for scale mapping branch (size = g_size*scale_mapping_max)', type=click.IntRange(min=0))
@click.option('--scale_mapping_norm', help='normalization type for scale mapping branch', type=click.Choice(['positive', 'zerocentered']), default='positive')

# additional options for 360 model
@click.option('--fov', help='fov for one frame in the 360 model', type=click.IntRange(min=0), default=60, show_default=True)
# 4096 x 4096 base view
@click.option('--use_hr', help='use high resolution base view', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--actual_res', help='actual resolution of the base view, if not given, use ', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--use_scale_on_top', help='Should we use scale on top of patch way, or disable it?', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--ds_mode', help='downsampling mode', type=click.Choice(['nearest', 'bilinear', 'average', 'bicubic']), default='average', show_default=True)
@click.option('--l2_lambda', help='l2 loss weight', metavar='FLOAT', type=click.FloatRange(min=0), default=0.0, show_default=True)
@click.option('--use_grad', help='use grad for patch training adverserial loss', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--scale_grad', help="scale grad for patch training, between 0 and 1", metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--use_normal_x', help='use normal position sampling x for patch training adverserial loss', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--use_old_filters', help='use old compatible filters, else just use 4k ones', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--log_hr', help='log HR 1 Forward Pass images', is_flag=True, default=False)
@click.option('--log_imgs', help='log fakes and real imgs', is_flag=True, default=False)
@click.option('--log_lr', help='log LR reconfigured forward pass images', is_flag=True, default=False)
@click.option('--use_grad_clip', help='use grad clip for patch training adverserial loss', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--reinitd', help='reinit Discriminator', metavar='FLOAT', is_flag=True, default=False)
@click.option('--freezeg', help='Freeze first layers of G', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--deltag', help='Delta G', metavar='FLOAT', type=click.IntRange(min=0), default=0.0, show_default=True)
@click.option('--overrided', help='Override D, location of pkl file', metavar='[PATH|URL]', type=str)
@click.option('--bcond', help='Use bcond for D', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--bcondg', help='Use bcond for G', metavar='BOOL', type=bool, default=False, show_default=True)
def main(**kwargs):

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), freezeG=opts.freezeg, deltaG=opts.deltag, use_scale_on_top=opts.use_scale_on_top)
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8, weight_decay=opts.l2_lambda)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8, weight_decay=opts.l2_lambda)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror
    if opts.data_max_size:
        c.training_set_kwargs.max_size = opts.max_size
    if opts.data_resolution:
        if c.training_set_kwargs.resolution != opts.data_resolution:
            print("using specified data resolution %d rather than default" % (opts.data_resolution))
            c.training_set_kwargs.resolution = opts.data_resolution
    c.training_set_kwargs.crop_image = opts.random_crop
    # by this point, resolution should be determined
    # either from init_dataset function or opts.data_resolution
    assert(c.training_set_kwargs.resolution is not None)

    # set up training mode
    training_mode = c.G_kwargs.training_mode = opts.training_mode
    # set up generator size
    if opts.g_size is not None:
        assert(opts.g_size == c.training_set_kwargs.resolution)
    else:
        opts.g_size = c.training_set_kwargs.resolution
    if 'patch' in training_mode:
        # patch dataset kwargs
        patch_kwargs = dnnlib.EasyDict(
            class_name='training.dataset.ImagePatchDataset',
            path=opts.data_hr, resolution=opts.g_size if opts.patch_size ==0 else opts.patch_size,
            scale_min=opts.scale_min, scale_max=opts.scale_max,
            scale_anneal=opts.scale_anneal, random_crop=opts.patch_crop,
            use_labels=True, max_size=None, xflip=False, use_normal=opts.use_normal_x, use_hr=opts.use_hr)
        patch_obj = dnnlib.util.construct_class_by_name(**patch_kwargs) # gets initial args
        patch_name = patch_obj.name
        patch_kwargs.resolution = patch_obj.resolution # Be explicit about resolution.
        patch_kwargs.use_labels = patch_obj.has_labels # Be explicit about labels.
        patch_kwargs.max_size = len(patch_obj) # Be explicit about dataset size.
        c.patch_kwargs = patch_kwargs
        c.patch_kwargs.use_labels = opts.cond
        c.patch_kwargs.xflip = opts.mirror
        if opts.data_hr_max_size:
            c.patch_kwargs.max_size = opts.data_hr_max_size
        # added G_kwargs
        c.G_kwargs.scale_mapping_kwargs = dnnlib.EasyDict(
            scale_mapping_min = opts.scale_mapping_min,
            scale_mapping_max = opts.scale_mapping_max,
            scale_mapping_norm = opts.scale_mapping_norm
        )
        # added training options
        c.added_kwargs = dnnlib.EasyDict(
            img_size=opts.g_size,
            teacher=opts.teacher,
            teacher_lambda=opts.teacher_lambda,
            teacher_mode=opts.teacher_mode,
            scale_min=opts.scale_min,
            scale_max=opts.scale_max,
            scale_anneal=opts.scale_anneal,
            base_probability=opts.base_probability,
            use_hr=opts.use_hr, # Added
            use_grad=opts.use_grad, # Added
            scale_grad=opts.scale_grad, # Added
            use_teached_layers = ['synthesis.L1_36_1024.down_filter', 'synthesis.L2_52_1024.up_filter', 'synthesis.L2_52_1024.down_filter', 'synthesis.L3_52_1024.down_filter', 'synthesis.L4_84_1024.up_filter', 'synthesis.L4_84_1024.down_filter', 'synthesis.L5_148_1024.down_filter', 'synthesis.L7_276_645.down_filter', 'synthesis.L8_276_406.down_filter', 'synthesis.L9_532_256.down_filter', 'synthesis.L10_1044_161.up_filter', 'synthesis.L10_1044_161.down_filter', 'synthesis.L11_1044_102.down_filter', 'synthesis.L12_1044_64.up_filter'],
            actual_resolution=opts.actual_res if opts.actual_res > 0 else opts.g_size, # Added
            use_normal=opts.use_normal_x, # Added, though mainly used in PatchTrainingKwargs
            use_old_filters=opts.use_old_filters, # Added
            log_HR=opts.log_hr, # Added
            log_LR=opts.log_lr, # Added
            use_grad_clip = opts.use_grad_clip, # Added
            reinitd = opts.reinitd, # Added
            overrided = opts.overrided, # Added
            bcond = opts.bcond, # Added
            log_imgs = opts.log_imgs, # Added
            bcondg = opts.bcondg, # Added
        )
        if opts.use_hr:
            c.G_kwargs.use_scale_affine = False # TODO:for now disable scaling totally
    elif '360' in training_mode:
        c.G_kwargs.fov = opts.fov

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = (8 if opts.cfg == 'stylegan2' else 2) if opts.map_depth is None else opts.map_depth
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.loss_kwargs.r1_gamma = opts.gamma
    c.loss_kwargs.use_scale_on_top = opts.use_scale_on_top
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    if 'patch' in training_mode:
        c.patch_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers
    
    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    if opts.cfg == 'stylegan2':
        c.G_kwargs.class_name = 'training.networks_stylegan2.Generator'
        c.loss_kwargs.style_mixing_prob = 0.9 # Enable style mixing regularization.
        c.loss_kwargs.pl_weight = 2 # Enable path length regularization.
        c.G_reg_interval = 4 # Enable lazy regularization for G.
        c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
        c.loss_kwargs.pl_no_weight_grad = True # Speed up path length regularization by skipping gradient computation wrt. conv2d weights.
    else:
        c.G_kwargs.class_name = 'training.networks_stylegan3.Generator'
        c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
        if opts.cfg == 'stylegan3-r':
            c.G_kwargs.conv_kernel = 1 # Use 1x1 convolutions.
            c.G_kwargs.channel_base *= 2 # Double the number of feature maps.
            c.G_kwargs.channel_max *= 2
            c.G_kwargs.use_radial_filters = True # Use radially symmetric downsampling filters.
            c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
            c.loss_kwargs.blur_fade_kimg = c.batch_size * 200 / 32 # Fade out the blur during the first N kimg.

    if opts.use_hr and opts.bcondg:
        c.G_reg_interval = 1 # change the teacher model and student model to use different conditioning

    # Augmentation.
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume

    if opts.teacher is not None or opts.resume is not None:
        # disable rampups for finetuning or resuming models
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.
    if opts.use_hr:
        c.loss_kwargs.ds_mode = opts.ds_mode # added to support different ds modes
        #c.G_kwargs.skip_list = ["synthesis.L1_36_1024.down_filter", "synthesis.L2_52_1024.up_filter", "synthesis.L2_52_1024.down_filter", "synthesis.L3_52_1024.down_filter", "synthesis.L4_84_1024.up_filter", "synthesis.L4_84_1024.down_filter", "synthesis.L5_148_1024.down_filter", "synthesis.L7_276_645.down_filter", "synthesis.L8_276_406.down_filter", "synthesis.L9_532_256.down_filter", "synthesis.L10_1044_161.up_filter", "synthesis.L10_1044_161.down_filter", "synthesis.L11_1044_102.down_filter", "synthesis.L12_1044_64.up_filter"]
        #c.G_kwargs.skip_list = ["synthesis.L1_36_1024.down_filter", "synthesis.L2_52_1024.up_filter", "synthesis.L2_52_1024.down_filter", "synthesis.L3_52_1024.down_filter", "synthesis.L4_84_1024.up_filter", "synthesis.L4_84_1024.down_filter", "synthesis.L5_148_1024.down_filter", "synthesis.L7_276_645.down_filter", "synthesis.L8_276_406.down_filter", "synthesis.L9_532_256.down_filter", "synthesis.L10_1044_161.up_filter", "synthesis.L10_1044_161.down_filter", "synthesis.L11_1044_102.down_filter", "synthesis.L12_1044_64.up_filter"]
    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}--teacher{opts.teacher_lambda:.1f}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
