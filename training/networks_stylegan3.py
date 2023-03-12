# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generator architecture from the paper
"Alias-Free Generative Adversarial Networks"."""

import numpy as np
import scipy.signal
import scipy.optimize
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import filtered_lrelu
from torch_utils.ops import bias_act
import math
import random

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                  # Input tensor: [batch_size, in_channels, in_height, in_width]
    w,                  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
    s,                  # Style tensor: [batch_size, in_channels]
    demodulate  = True, # Apply weight demodulation?
    padding     = 0,    # Padding: int or [padH, padW]
    input_gain  = None, # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
):
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape
    misc.assert_shape(w, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(s, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs.
    if demodulate:
        w = w * w.square().mean([1,2,3], keepdim=True).rsqrt()
        s = s * s.square().mean().rsqrt()

    # Modulate weights.
    w = w.unsqueeze(0) # [NOIkk]
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Demodulate weights.
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
        w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Apply input scaling.
    if input_gain is not None:
        input_gain = input_gain.expand(batch_size, in_channels) # [NI]
        w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_gradfix.conv2d(input=x, weight=w.to(x.dtype), padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        bias            = True,     # Apply additive bias before the activation function?
        lr_multiplier   = 1,        # Learning rate multiplier.
        weight_init     = 1,        # Initial standard deviation of the weight tensor.
        bias_init       = 0,        # Initial value of the additive bias.
        frozen          = False,    # Whether to freeze the layer.
        use_delta       = False,    # Whether to use delta weights
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        if not frozen:
            self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
            bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
            self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        else:
            self.register_buffer('weight', torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
            bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
            self.register_buffer('bias', torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier
        self.use_delta = use_delta
        if use_delta:
            self.weight_delta = torch.nn.Parameter(torch.zeros([out_features, in_features]))
            self.bias_delta = torch.nn.Parameter(torch.zeros([out_features])) if bias else None
    
    def remove_delta_weights(self):
        self.use_delta = False
        self.weight_delta = None
        self.bias_delta = None

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain if not self.use_delta else self.weight.to(x.dtype) * self.weight_gain + self.weight_delta.to(x.dtype)
        b = self.bias if not self.use_delta else self.bias + self.bias_delta
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}, use_delta={self.use_delta}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no labels.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output.
        num_layers      = 2,        # Number of mapping layers.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training.
        frozen          = False,    # Whether to freeze the mapping layers.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        # Construct layers.
        self.embed = FullyConnectedLayer(self.c_dim, self.w_dim, frozen=frozen) if self.c_dim > 0 else None
        features = [self.z_dim + (self.w_dim if self.c_dim > 0 else 0)] + [self.w_dim] * self.num_layers
        for idx, in_features, out_features in zip(range(num_layers), features[:-1], features[1:]):
            layer = FullyConnectedLayer(in_features, out_features, activation='lrelu', lr_multiplier=lr_multiplier, frozen=frozen)
            setattr(self, f'fc{idx}', layer)
        self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        misc.assert_shape(z, [None, self.z_dim])
        if truncation_cutoff is None:
            truncation_cutoff = self.num_ws

        # Embed, normalize, and concatenate inputs.
        x = z.to(torch.float32)
        x = x * (x.square().mean(1, keepdim=True) + 1e-8).rsqrt()
        if self.c_dim > 0:
            misc.assert_shape(c, [None, self.c_dim])
            y = self.embed(c.to(torch.float32))
            y = y * (y.square().mean(1, keepdim=True) + 1e-8).rsqrt()
            x = torch.cat([x, y], dim=1) if x is not None else y

        # Execute layers.
        for idx in range(self.num_layers):
            x = getattr(self, f'fc{idx}')(x)

        # Update moving average of W.
        if update_emas:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast and apply truncation.
        x = x.unsqueeze(1).repeat([1, self.num_ws, 1])
        if truncation_psi != 1:
            x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisInput(torch.nn.Module):
    def __init__(self,
        w_dim,          # Intermediate latent (W) dimensionality.
        channels,       # Number of output channels.
        size,           # Output spatial size: int or [width, height].
        sampling_rate,  # Output sampling rate.
        bandwidth,      # Output bandwidth.
        margin_size,    # Extra margin on input.
        frozen = False, # Whether to freeze the parameters.
        delta  = False, # Whether to use delta parameters.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.channels = channels
        self.size = np.broadcast_to(np.asarray(size), [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth
        self.margin_size = margin_size

        # Draw random frequencies from uniform 2D disc.
        freqs = torch.randn([self.channels, 2])
        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= bandwidth
        phases = torch.rand([self.channels]) - 0.5

        # Setup parameters and buffers.
        if not frozen:
            self.weight = torch.nn.Parameter(torch.randn([self.channels, self.channels]))
        else:
            self.register_buffer('weight', torch.randn([self.channels, self.channels]))
        self.affine = FullyConnectedLayer(w_dim, 4, weight_init=0, bias_init=[1,0,0,0], frozen=frozen)
        self.register_buffer('transform', torch.eye(3, 3)) # User-specified inverse transform wrt. resulting image.
        self.register_buffer('freqs', freqs)
        self.register_buffer('phases', phases)

    def forward(self, w, transform=None, **kwargs):
        # Introduce batch dimension.
        if transform is None:
            # sanity check; should not modify transform from identity
            assert(torch.equal(self.transform, torch.eye(3, 3).to(self.transform.device)))
            transform = self.transform.unsqueeze(0) # [batch, row, col]
        freqs = self.freqs.unsqueeze(0) # [batch, channel, xy]
        phases = self.phases.unsqueeze(0) # [batch, channel]

        # Apply learned transformation.
        t = self.affine(w) # t = (r_c, r_s, t_x, t_y)
        t = t / t[:, :2].norm(dim=1, keepdim=True) # t' = (r'_c, r'_s, t'_x, t'_y)
        m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse rotation wrt. resulting image.
        m_r[:, 0, 0] = t[:, 0]  # r'_c
        m_r[:, 0, 1] = -t[:, 1] # r'_s
        m_r[:, 1, 0] = t[:, 1]  # r'_s
        m_r[:, 1, 1] = t[:, 0]  # r'_c
        m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse translation wrt. resulting image.
        m_t[:, 0, 2] = -t[:, 2] # t'_x
        m_t[:, 1, 2] = -t[:, 3] # t'_y
        transforms = m_r @ m_t @ transform # First rotate resulting image, then translate, and finally apply user-specified transform.

        # Transform frequencies.
        phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transforms[:, :2, :2]
        # Dampen out-of-band frequencies that may occur due to the user-specified transform.
        amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)

        # Construct sampling grid.
        theta = torch.eye(2, 3, device=w.device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False)

        # print(f'{grids.shape=} | {transforms.shape=}') # Added, checking if continues grid sampling would work?
        # grids = grids[:, :18, :18, :].permute(0, 3, 1, 2) # [batch, channel, height, width]
        # interpolate with torch functional
        # grids = torch.nn.functional.interpolate(grids, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3, 1) # [batch, height, width, channel]
        # Compute Fourier features.
        x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) # [batch, height, width, channel]
        x = x + phases.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (np.pi * 2))
        # print(f"{x.shape=} * {amplitudes.shape=}")
        x = x * amplitudes.unsqueeze(1).unsqueeze(2)

        # Apply trainable mapping.
        weight = self.weight / np.sqrt(self.channels)
        # print(f"{x.shape=} @ {weight.t().shape=}")
        # only on first and second dim
        
        x = x @ weight.t()
        # print(f"{x.shape=} after @")
        

        # Ensure correct shape.
        x = x.permute(0, 3, 1, 2) # [batch, channel, height, width]
        # print(f"{x.shape=}")
        
        #misc.assert_shape(x, [w.shape[0], self.channels, int(self.size[1]), int(self.size[0])])
        return x
    def reconfigure_network(self, size=None, sampling_rate=None, bandwidth=None):
        """ Reconfigure the network for a different output size, sampling rate, or bandwidth.

        Args:
            size (_type_, optional): _description_. Defaults to None.
            sampling_rate (_type_, optional): _description_. Defaults to None.
            bandwidth (_type_, optional): _description_. Defaults to None.
        """        
        if size is not None:
            self.size = np.broadcast_to(np.asarray(size), [2])
        if sampling_rate is not None:
            self.sampling_rate = sampling_rate
        if bandwidth is not None:
            self.bandwidth = bandwidth

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, channels={self.channels:d}, size={list(self.size)},',
            f'sampling_rate={self.sampling_rate:g}, bandwidth={self.bandwidth:g}'])


@persistence.persistent_class
class SynthesisInput360(torch.nn.Module):
    def __init__(self,
        w_dim,          # Intermediate latent (W) dimensionality.
        channels,       # Number of output channels.
        size,           # Output spatial size: int or [width, height].
        sampling_rate,  # Output sampling rate.
        bandwidth,      # Output bandwidth.
        margin_size,    # Extra margin on input.
        fov,            # panorama FOV.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.channels = channels
        self.fov = fov
        self.size = np.broadcast_to(np.asarray(size), [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth
        self.margin_size = margin_size
        self.frame_size = self.size - 2 * self.margin_size

        # Draw random frequencies from uniform 2D disc.
        freqs = torch.randn([self.channels, 2])
        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= bandwidth
        phases = torch.rand([self.channels]) - 0.5

        # Setup parameters and buffers.
        self.weight = torch.nn.Parameter(torch.randn([self.channels, self.channels]))
        self.affine = FullyConnectedLayer(w_dim, 4, weight_init=0, bias_init=[1,0,0,0])
        self.register_buffer('transform', torch.eye(3, 3)) # User-specified inverse transform wrt. resulting image.
        self.register_buffer('freqs', freqs)
        self.register_buffer('phases', phases)

    def forward(self, w, transform=None, crop_fn=None):
        # Introduce batch dimension.
        if transform is None:
            transforms = self.transform.unsqueeze(0) # [batch, row, col]
        else:
            transforms = transform
        freqs = self.freqs.unsqueeze(0) # [batch, channel, xy]
        phases = self.phases.unsqueeze(0) # [batch, channel]

        # does not add learned rotation for 360 model
        transforms = transforms.expand(w.shape[0], -1, -1)

        # Dampen out-of-band frequencies that may occur due to the user-specified transform.
        amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)

        # Construct sampling grid.
        theta = torch.eye(2, 3, device=w.device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate # tx
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate # ty
        grid_width = self.frame_size[0] * 360 // self.fov + 2 * self.margin_size
        grids = torch.nn.functional.affine_grid(theta.unsqueeze(0),
                                                [1, 1, self.size[1], grid_width],
                                                align_corners=False)
        # extended grid to ensure that the x coordinate completes a full circle without padding
        base_width = grid_width - 2*self.margin_size
        corrected_x = torch.arange(-self.margin_size, base_width*2+self.margin_size, device=grids.device) / base_width  * 2 - 1
        corrected_y = grids[0, :, 0, 1]
        corrected_grids = torch.cat([corrected_x.view(1, 1, -1, 1).repeat(1, self.size[1], 1, 1),
                                     corrected_y.view(1, -1, 1, 1).repeat(1, 1, grid_width+base_width, 1)], dim=3)
        grids = corrected_grids

        if crop_fn is None:
            crop_start = random.randint(0, base_width - 1)
            grids = grids[:, :, crop_start:crop_start+self.size[1], :]
        else:
            grids = crop_fn(grids)

        # apply transformation first
        rotation = transforms[:, :2, :2]
        translation = transforms[:, :2, 2:].squeeze(2)
        # normalize grid x s.t. transformations can operate on square affine ratio
        grids_normalized = grids.clone()
        min_bound = torch.min(grids_normalized[:, :, :, 0])
        max_bound = torch.max(grids_normalized[:, :, :, 0])
        target_range = torch.max(grids_normalized[:, :, :, 1]) - torch.min(grids_normalized[:, :, :, 1])
        grids_normalized[:, :, :, 0] = (grids_normalized[:, :, :, 0] - min_bound) / (max_bound - min_bound)
        grids_normalized[:, :, :, 0] = grids_normalized[:, :, :, 0] * target_range - target_range / 2
        # # xT @ RT = (Rx)T --> it is transposed
        grids_transformed = (grids_normalized.unsqueeze(3) @ rotation.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3)
        grids_transformed = grids_transformed + translation.unsqueeze(1).unsqueeze(2)
        grids_transformed[:, :, :, 0] = (grids_transformed[:, :, :, 0] + target_range / 2) / target_range * (max_bound - min_bound) + min_bound

        # map discontinuous x-angle to continuous cylindrical coordinate
        grids_transformed_sin = grids_transformed.clone()
        grids_transformed_cos = grids_transformed.clone()
        grids_transformed_sin[:, :, :, 0] = torch.sin(grids_transformed_sin[:, :, :, 0] * torch.tensor(math.pi))
        grids_transformed_cos[:, :, :, 0] = torch.cos(grids_transformed_cos[:, :, :, 0] * torch.tensor(math.pi))

        x_sin = (grids_transformed_sin.unsqueeze(3) @ freqs[:, :self.channels//2, :].permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) # [batch, height, width, channel]
        x_sin = x_sin + phases[:, :self.channels//2].unsqueeze(1).unsqueeze(2)
        x_sin = torch.sin(x_sin * (np.pi * 2))
        x_sin = x_sin * amplitudes[:, :self.channels//2].unsqueeze(1).unsqueeze(2)
        x_cos = (grids_transformed_cos.unsqueeze(3) @ freqs[:, self.channels//2:, :].permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) # [batch, height, width, channel]
        x_cos = x_cos + phases[:, self.channels//2:].unsqueeze(1).unsqueeze(2)
        x_cos = torch.sin(x_cos * (np.pi * 2))
        x_cos = x_cos * amplitudes[:, self.channels//2:].unsqueeze(1).unsqueeze(2)
        x = torch.cat([x_sin, x_cos], dim=-1)

        # Apply trainable mapping.
        weight = self.weight / np.sqrt(self.channels)
        x = x @ weight.t()

        # Ensure correct shape.
        x = x.permute(0, 3, 1, 2) # [batch, channel, height, width]
        misc.assert_shape(x, [w.shape[0], self.channels, int(self.size[1]), int(self.size[0])])
        return x


#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        w_dim,                          # Intermediate latent (W) dimensionality.
        is_torgb,                       # Is this the final ToRGB layer?
        is_critically_sampled,          # Does this layer use critical sampling?
        use_fp16,                       # Does this layer use FP16?

        # Input & output specifications.
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        in_sampling_rate,               # Input sampling rate (s).
        out_sampling_rate,              # Output sampling rate (s).
        in_cutoff,                      # Input cutoff frequency (f_c).
        out_cutoff,                     # Output cutoff frequency (f_c).
        in_half_width,                  # Input transition band half-width (f_h).
        out_half_width,                 # Output Transition band half-width (f_h).

        # Hyperparameters.
        conv_kernel         = 3,        # Convolution kernel size. Ignored for final the ToRGB layer.
        filter_size         = 6,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        lrelu_upsampling    = 2,        # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
        use_radial_filters  = False,    # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
        conv_clamp          = 256,      # Clamp the output to [-X, +X], None = disable clamping.
        magnitude_ema_beta  = 0.999,    # Decay rate for the moving average of input magnitudes. 

        # added
        use_scale_affine    = False,
        frozen              = False,    # Freeze the weights?
        delta               = False,    # Whether to use delta weights on the affine layer (style)
    ):
        super().__init__()
        self.lrelu_upsampling = lrelu_upsampling # added so recofig can be used
        self.filter_size = filter_size # added so recofig can be used
        self.use_radial_filters = use_radial_filters # added so recofig can be used
        self.w_dim = w_dim
        self.is_torgb = is_torgb
        self.is_critically_sampled = is_critically_sampled
        self.use_fp16 = use_fp16
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.conv_kernel = 1 if is_torgb else conv_kernel
        self.conv_clamp = conv_clamp
        self.magnitude_ema_beta = magnitude_ema_beta
        self.frozen = frozen
        self.delta = delta
        # Setup parameters and buffers.
        self.affine = FullyConnectedLayer(self.w_dim, self.in_channels, bias_init=1, frozen=frozen, use_delta=delta)
        if frozen:
            self.weight = torch.nn.Parameter(torch.randn([self.out_channels, self.in_channels, self.conv_kernel, self.conv_kernel]))
            self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        else:
            self.register_buffer('weight', torch.randn([self.out_channels, self.in_channels, self.conv_kernel, self.conv_kernel]))
            self.register_buffer('bias', torch.zeros([self.out_channels]))
        self.register_buffer('magnitude_ema', torch.ones([]))

        # Design upsampling filter.
        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.up_factor == self.tmp_sampling_rate
        self.up_taps = filter_size * self.up_factor if self.up_factor > 1 and not self.is_torgb else 1
        #print(dict(numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate))
        self.register_buffer('up_filter', self.design_lowpass_filter(
            numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate))

        # Design downsampling filter.
        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.down_factor == self.tmp_sampling_rate
        self.down_taps = filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
        self.down_radial = use_radial_filters and not self.is_critically_sampled
        self.register_buffer('down_filter', self.design_lowpass_filter(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate, radial=self.down_radial))

        # Compute padding.
        pad_total = (self.out_size - 1) * self.down_factor + 1 # Desired output size before downsampling.
        pad_total -= (self.in_size + self.conv_kernel - 1) * self.up_factor # Input size after upsampling.
        pad_total += self.up_taps + self.down_taps - 2 # Size reduction caused by the filters.
        pad_lo = (pad_total + self.up_factor) // 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

        # added
        self.use_scale_affine = use_scale_affine
        if self.use_scale_affine:
            self.scale_affine = FullyConnectedLayer(self.w_dim, self.in_channels, bias_init=0)

    def reconfigure_network(self, in_size, out_size, in_cutoff, out_cutoff, in_sampling_rate, out_sampling_rate, in_half_width,
                            out_half_width, use_fp16, is_critically_sampled):
        device = self.weight.device
        self.use_fp16 = use_fp16
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.is_critically_sampled = is_critically_sampled
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if self.is_torgb else self.lrelu_upsampling)

        self.in_half_width = in_half_width
        self.out_half_width = out_half_width

        # Design upsampling filter.
        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.up_factor == self.tmp_sampling_rate
        self.up_taps = self.filter_size * self.up_factor if self.up_factor > 1 and not self.is_torgb else 1
        # print(f"{self.up_taps=}, {self.in_cutoff=}, {self.in_half_width=}, {self.tmp_sampling_rate=}")
        self.down_radial = self.use_radial_filters and not self.is_critically_sampled
        #print(dict(numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate))
        lowpass_filter = self.design_lowpass_filter(numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate)
        self.register_buffer('up_filter', lowpass_filter.to(device) if lowpass_filter is not None else lowpass_filter) # last one is None (identity)
        
        # Design downsampling filter.
        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.down_factor == self.tmp_sampling_rate
        self.down_taps = self.filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
        self.down_radial = self.use_radial_filters and not self.is_critically_sampled # TODO: think do we need to change it? probably not
        lowpass_filter = self.design_lowpass_filter(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate, radial=self.down_radial)
        self.register_buffer("down_filter", lowpass_filter.to(device) if lowpass_filter is not None else lowpass_filter) # last one is None (identity)

        # Compute padding.
        pad_total = (self.out_size - 1) * self.down_factor + 1
        pad_total -= (self.in_size + self.conv_kernel - 1) * self.up_factor
        pad_total += self.up_taps + self.down_taps - 2
        pad_lo = (pad_total + self.up_factor) // 2
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

    def get_up_filter(self, in_size, out_size, in_cutoff, out_cutoff, in_sampling_rate, out_sampling_rate, in_half_width, out_half_width, use_fp16, is_critically_sampled):
        tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if self.is_torgb else self.lrelu_upsampling)
        down_taps = self.filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
        down_radial = self.use_radial_filters and not self.is_critically_sampled # TODO: think do we need to change it? probably not
        return self.design_lowpass_filter(numtaps=down_taps, cutoff=out_cutoff, width=out_half_width*2, fs=self.tmp_sampling_rate, radial=down_radial)

    def get_down_filter(self, in_size, out_size, in_cutoff, out_cutoff, in_sampling_rate, out_sampling_rate, in_half_width,
                            out_half_width, use_fp16, is_critically_sampled):
        tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if self.is_torgb else self.lrelu_upsampling)
        down_taps = self.filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
        down_radial = self.use_radial_filters and not self.is_critically_sampled # TODO: think do we need to change it? probably not
        return self.design_lowpass_filter(
            numtaps=down_taps, cutoff=out_cutoff, width=out_half_width*2, fs=tmp_sampling_rate, radial=self.down_radial)

    def forward(self, x, w, scale=None, noise_mode='random', force_fp32=False, update_emas=False):
        assert noise_mode in ['random', 'const', 'none'] # unused
        #misc.assert_shape(x, [None, self.in_channels, int(self.in_size[1]), int(self.in_size[0])])
        misc.assert_shape(w, [x.shape[0], self.w_dim])

        # Track input magnitude.
        if update_emas:
            with torch.autograd.profiler.record_function('update_magnitude_ema'):
                magnitude_cur = x.detach().to(torch.float32).square().mean()
                self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.magnitude_ema_beta))
        input_gain = self.magnitude_ema.rsqrt()

        # Execute affine layer.
        styles = self.affine(w)
        # added here
        if self.use_scale_affine:
            assert(scale is not None)
            styles_scale = self.scale_affine(scale)
            styles = styles + styles_scale # equivalent to concatenation

        if self.is_torgb:
            weight_gain = 1 / np.sqrt(self.in_channels * (self.conv_kernel ** 2))
            styles = styles * weight_gain

        # Execute modulated conv2d.
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        x = modulated_conv2d(x=x.to(dtype), w=self.weight, s=styles,
            padding=self.conv_kernel-1, demodulate=(not self.is_torgb), input_gain=input_gain)

        # Execute bias, filtered leaky ReLU, and clamping.
        gain = 1 if self.is_torgb else np.sqrt(2)
        slope = 1 if self.is_torgb else 0.2
        x = filtered_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
            up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=self.conv_clamp)

        # Ensure correct shape and dtype.
        # misc.assert_shape(x, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])])
        assert x.dtype == dtype
        return x

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
        assert numtaps >= 1

        # Identity filter.
        if numtaps == 1:
            return None

        # Separable Kaiser low-pass filter.
        if not radial:
            f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return torch.as_tensor(f, dtype=torch.float32)

        # Radially symmetric jinc-based filter.
        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return torch.as_tensor(f, dtype=torch.float32)

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, is_torgb={self.is_torgb},',
            f'is_critically_sampled={self.is_critically_sampled}, use_fp16={self.use_fp16},',
            f'in_sampling_rate={self.in_sampling_rate:g}, out_sampling_rate={self.out_sampling_rate:g},',
            f'in_cutoff={self.in_cutoff:g}, out_cutoff={self.out_cutoff:g},',
            f'in_half_width={self.in_half_width:g}, out_half_width={self.out_half_width:g},',
            f'in_size={list(self.in_size)}, out_size={list(self.out_size)},',
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, frozen={self.frozen}, delta={self.delta}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                          # Intermediate latent (W) dimensionality.
        img_resolution,                 # Output image resolution.
        img_channels,                   # Number of color channels.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_layers          = 14,       # Total number of layers, excluding Fourier features and ToRGB.
        num_critical        = 2,        # Number of critically sampled layers at the end.
        first_cutoff        = 2,        # Cutoff frequency of the first layer (f_{c,0}).
        first_stopband      = 2**2.1,   # Minimum stopband of the first layer (f_{t,0}).
        last_stopband_rel   = 2**0.3,   # Minimum stopband of the last layer, expressed relative to the cutoff.
        margin_size         = 10,       # Number of additional pixels outside the image.
        output_scale        = 0.25,     # Scale factor for the output image.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        training_mode       = 'global', # training mode for input layer
        fov                 = None,     # Specify FOV for 360 model
        actual_resolution   = 1024,     # Specify actual resolution for size calculation
        freezeG             = 0,        # freeze N layers of generator
        deltaG              = 0,        # use delta weight for affine of last N layers of generator, also free them if so
        **layer_kwargs,                 # Arguments for SynthesisLayer.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.num_ws = num_layers + 2
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_layers = num_layers
        self.num_critical = num_critical
        self.margin_size = margin_size
        self.output_scale = output_scale
        self.num_fp16_res = num_fp16_res
        self.channel_max = channel_max # saved for later use
        self.channel_base = channel_base
        self.reconfigure_back_layers = None # default
        # Geometric progression of layer cutoffs and min. stopbands.
        last_cutoff = self.img_resolution / 2 # f_{c,N}
        last_stopband = last_cutoff * last_stopband_rel # f_{t,N}
        exponents = np.minimum(np.arange(self.num_layers + 1) / (self.num_layers - self.num_critical), 1)
        cutoffs = first_cutoff * (last_cutoff / first_cutoff) ** exponents # f_c[i]
        stopbands = first_stopband * (last_stopband / first_stopband) ** exponents # f_t[i]

        # Compute remaining layer parameters.
        sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands * 2, self.img_resolution)))) # s[i]
        half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs # f_h[i]
        sizes = sampling_rates + self.margin_size * 2
        sizes[-2:] = self.img_resolution
        channels = np.rint(np.minimum((channel_base / 2) / cutoffs, channel_max))
        channels[-1] = self.img_channels

        channels, sizes, sampling_rates, cutoffs, half_widths = self.compute_stuff_for_resolution(img_resolution, channel_base, channel_max)
        channels_1k, sizes_1k, sampling_rates_1k, cutoffs_1k, half_widths_1k = self.compute_stuff_for_resolution(actual_resolution, channel_base, channel_max)

        def trainable_gen(num, max_items, rev=False):
            items = [1 if i < num else 0 for i in range(max_items + 1)] 
            items = items if not rev else reversed(items)
            for i in items:
                yield i == 1
            raise ValueError('too many items requested')
        
        frozen_iter = trainable_gen(freezeG, num_layers)
        delta_iter = trainable_gen(deltaG, num_layers, True)
        # Construct layers.
        if '360' not in training_mode:
            self.input = SynthesisInput(
                w_dim=self.w_dim, channels=int(channels[0]), size=int(sizes[0]),
                sampling_rate=sampling_rates[0], bandwidth=cutoffs[0],
                margin_size=margin_size, frozen=freezeG > 0)
        else:
            assert(fov is not None)
            self.input = SynthesisInput360(
                w_dim=self.w_dim, channels=int(channels[0]), size=int(sizes[0]),
                sampling_rate=sampling_rates[0], bandwidth=cutoffs[0],
                margin_size=margin_size, fov=fov)
        self.layer_names = []
        for idx in range(self.num_layers + 1):
            prev = max(idx - 1, 0)
            is_torgb = (idx == self.num_layers)
            is_critically_sampled = (idx >= self.num_layers - self.num_critical)
            use_fp16 = (sampling_rates[idx] * (2 ** self.num_fp16_res) > self.img_resolution)
            delta, frozen = next(delta_iter), next(frozen_iter)
            frozen = frozen or delta
            if actual_resolution != img_resolution: # Added
                print(f'Reminder: using all at {img_resolution} except in_channels={int(channels_1k[prev])} and out_channels={int(channels_1k[idx])}')
            layer = SynthesisLayer( # changed in_channels to reflex 1k always
                w_dim=self.w_dim, is_torgb=is_torgb, is_critically_sampled=is_critically_sampled, use_fp16=use_fp16,
                in_channels=int(channels_1k[prev]), out_channels=int(channels_1k[idx]),
                in_size=int(sizes[prev]), out_size=int(sizes[idx]),
                in_sampling_rate=int(sampling_rates[prev]), out_sampling_rate=int(sampling_rates[idx]),
                in_cutoff=cutoffs[prev], out_cutoff=cutoffs[idx],
                in_half_width=half_widths[prev], out_half_width=half_widths[idx], frozen=frozen, delta=delta,
                **layer_kwargs)
            #name = f'L{idx}_{layer.out_size[0]}_{layer.out_channels}'
            name = f'L{idx}_{int(sizes_1k[idx])}_{int(channels_1k[idx])}'
            setattr(self, name, layer)
            self.layer_names.append(name)

    def remove_all_delta_weights(self, rank):
        for layer_name in self.layer_names:
            layer: SynthesisLayer = getattr(self, layer_name)
            layer.affine.remove_delta_weights()
        if rank == 0:
            print('Removed all delta weights for teacher model')

    def compute_stuff_for_resolution(self, img_resolution, channel_base, channel_max): # Added
        first_cutoff        = 2       # Cutoff frequency of the first layer (f_{c,0}).
        first_stopband      = 2**2.1  # Minimum stopband of the first layer (f_{t,0}).
        last_stopband_rel   = 2**0.3  # Minimum stopband of the last layer, expressed relative to the cutoff.

        last_cutoff = img_resolution / 2 # f_{c,N}
        last_stopband = last_cutoff * last_stopband_rel # f_{t,N}
        exponents = np.minimum(np.arange(self.num_layers + 1) / (self.num_layers - self.num_critical), 1)
        cutoffs = first_cutoff * (last_cutoff / first_cutoff) ** exponents # f_c[i]
        stopbands = first_stopband * (last_stopband / first_stopband) ** exponents # f_t[i]

        # Compute remaining layer parameters.
        sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands * 2, img_resolution)))) # s[i]
        half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs # f_h[i]
        sizes = sampling_rates + self.margin_size * 2
        sizes[-2:] = img_resolution

        channels = np.rint(np.minimum((channel_base / 2) / cutoffs, channel_max))
        channels[-1] = self.img_channels
        # print(f"{cutoffs=}")
        return channels, sizes, sampling_rates, cutoffs, half_widths

    def reconfigure_network(self, img_resolution, channel_base = 32768 * 2, channel_max= 1024, use_old_filters:bool = True): # Added
        channels, sizes, sampling_rates, cutoffs, half_widths = self.compute_stuff_for_resolution(img_resolution, channel_base, channel_max)
        channels_1k, sizes_1k, sampling_rates_1k, cutoffs_1k, half_widths_1k = self.compute_stuff_for_resolution(1024, channel_base, channel_max)
        # reconfigure input (size, sampling rate and bandwidth)
        self.img_resolution = img_resolution
        self.input.reconfigure_network(size=int(sizes[0]), sampling_rate=int(sampling_rates[0]), bandwidth=cutoffs_1k[0])
        # iterate over synthesis layers and reconfigure them        
        for idx in range(self.num_layers + 1):
            prev = max(idx - 1, 0)
            name = f'L{idx}_{int(sizes_1k[idx])}_{int(channels_1k[idx])}'
            use_fp16 = (sampling_rates[idx] * (2 ** self.num_fp16_res) > self.img_resolution)
            is_critically_sampled = (idx >= self.num_layers - self.num_critical)
            # reconfigure only in_size, out_size, in_sampling_rate,     out_sampling_rate, in_half_width, out_half_width
            # if name not in change_layers:
            #     continue # skip, as they already configured correctly
            getattr(self, name).reconfigure_network(
                in_size=int(sizes[prev]), out_size=int(sizes[idx]),
                in_cutoff=cutoffs[prev], out_cutoff=cutoffs[idx],
                in_sampling_rate=int(sampling_rates[prev]), out_sampling_rate=int(sampling_rates[idx]),
                in_half_width=half_widths[prev], out_half_width=half_widths[idx], use_fp16=use_fp16, is_critically_sampled=is_critically_sampled
                )
        if self.reconfigure_back_layers and use_old_filters:
            device = self.input.transform.device
            for k, v in self.reconfigure_back_layers.items():
                key = k.replace('synthesis.',"")
                self.state_dict()[key].copy_(v).to(device)

    def add_reset_layers(self, dict_of_layers_and_value):
        self.reconfigure_back_layers = dict_of_layers_and_value
    
    def get_dict_of_up_filters(self): # Added
        def to_cpu(x):
            return x.cpu() if x is not None else None
        return {name:to_cpu(getattr(self, name).up_filter) for name in self.layer_names}
        
    def forward(self, ws, mapped_scale=None, transform=None, slice_range=None, mapped_slice = None, **layer_kwargs):
        # TODO: should we only use mapped_scale or mapped_slice? 
        misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        if isinstance(slice_range, tuple):
            slice_range = [slice_range] * ws.shape[0] # If using one slice, or a single slice for all images
        ws = ws.to(torch.float32).unbind(dim=1)
        if mapped_scale is not None:
            scale = mapped_scale.to(torch.float32).unbind(dim=1)
        else:
            scale = [None] * self.num_ws
        # ws is a list of ws for every layer
        # TODO: should we add a mapping like scale, for mapping_slice?
        

        # Execute layers.
        x = self.input(ws[0], transform=transform)
        
        if slice_range is not None: # Added to slice images
            assert slice_range.shape[0] == x.shape[0], f"{slice_range.shape[0]}!={x.shape[0]} slice_range must be None or have the same length as the batch size {slice_range.shape=}, {x.shape=}"
            #assert len(slice_range) == x.shape[0], f"{len(slice_range)}!={x.shape[0]} slice_range must be None or have the same length as the batch size"
            x = torch.stack([x[i, :, x_start:x_end, y_start:y_end] for i, (x_start, x_end, y_start, y_end) in zip(range(x.shape[0]), slice_range)])

        for name, w , sc in zip(self.layer_names, ws[1:], scale[1:]):
            x = getattr(self, name)(x, w, sc, **layer_kwargs)
        if self.output_scale != 1:
            x = x * self.output_scale

        # Ensure correct shape and dtype.
        # misc.assert_shape(x, [None, self.img_channels, self.img_resolution, self.img_resolution]) TODO: add exact check, for now with slicing will surely hit this
        # Added Crop the image, in case crop_fn
        # if crop_fn is not None and crop_params is not None and all([x is not None for x in crop_params]):
        #     x = crop_fn(x, crop_params)
            # misc.assert_shape(x, [None, self.img_channels, 1024, 1024]) # TODO: remove or parametrize this 
        x = x.to(torch.float32)
        return x

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_layers={self.num_layers:d}, num_critical={self.num_critical:d},',
            f'margin_size={self.margin_size:d}, num_fp16_res={self.num_fp16_res:d}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs       = {},  # Arguments for MappingNetwork.
        training_mode        = 'global',
        scale_mapping_kwargs = {},  # Arguments for Scale Mapping Network
        actual_resolution = None,
        freezeG              = 0,    # Number of frozen channels
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        self.actual_resolution = actual_resolution if actual_resolution is not None else img_resolution
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.training_mode = training_mode
        self.scale_mapping_kwargs = scale_mapping_kwargs
        self.use_scale_on_top = synthesis_kwargs.get('use_scale_on_top', True)
        self.use_scale_affine = True if 'patch' in self.training_mode else False # add affine layer on style input
        if not self.use_scale_on_top: # Added
            self.use_scale_affine = False # force not use affine layer on style input
        self.synthesis: SynthesisNetwork = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels,
                                          training_mode=training_mode, 
                                          actual_resolution=self.actual_resolution, freezeG = freezeG,
                                          **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, frozen = freezeG > 0, **mapping_kwargs)
        if 'patch' in self.training_mode: # Added to check if we actually use scale
            self.scale_mapping_kwargs = scale_mapping_kwargs
            scale_mapping_norm = scale_mapping_kwargs.scale_mapping_norm
            scale_mapping_min = scale_mapping_kwargs.scale_mapping_min
            scale_mapping_max = scale_mapping_kwargs.scale_mapping_max
            if scale_mapping_norm == 'zerocentered':
                self.scale_norm = ScaleNormalizeZeroCentered(scale_mapping_min, scale_mapping_max)
                scale_in_dim = 1
            elif scale_mapping_norm == 'positive':
                self.scale_norm = ScaleNormalizePositive(scale_mapping_min, scale_mapping_max)
                scale_in_dim = 1
            else:
                assert(False)
            self.scale_mapping = MappingNetwork(z_dim=scale_in_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)


    def forward(self, z, c, transform=None, slice_range=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        if transform is None:
            scale = torch.ones(z.shape[0], 1).to(z.device)
        else:
            scale = 1/transform[:, [0], 0]
        if self.scale_mapping_kwargs and self.actual_resolution == self.img_resolution and self.use_scale_on_top:
            scale = self.scale_norm(scale)
            mapped_scale = self.scale_mapping(scale, c, update_emas=update_emas)
        else:
            mapped_scale = None
        # TODO: maybe add own self.slice_mapping?
        img = self.synthesis(ws, mapped_scale=mapped_scale, slice_range=slice_range, transform=transform, update_emas=update_emas, **synthesis_kwargs)
        return img

    def reconfigure_network(self, img_resolution, channel_base:int = 32768* 2, channel_max:int = 1024, use_old_filters:bool = False):
        #print(f'Reconfiguring network from with params {img_resolution} and {channel_base} and {channel_max}')
        #print(f'Previous params are {self.img_resolution} and {self.synthesis.channel_base} and {self.synthesis.channel_max}')
        self.img_resolution = img_resolution
        self.synthesis.reconfigure_network(img_resolution=img_resolution, channel_base=channel_base, use_old_filters=use_old_filters)

#----------------------------------------------------------------------------
@persistence.persistent_class
class ScaleNormalizeZeroCentered(torch.nn.Module):
    def __init__(self, scale_mapping_min, scale_mapping_max):
        super().__init__()
        self.scale_mapping_min = scale_mapping_min
        self.scale_mapping_max = scale_mapping_max

    def forward(self, scale):
        # remaps scale to (-1, 1)
        scale = (scale - self.scale_mapping_min) / (self.scale_mapping_max - self.scale_mapping_min)
        return 2 * scale - 1

@persistence.persistent_class
class ScaleNormalizePositive(torch.nn.Module):
    def __init__(self, scale_mapping_min, scale_mapping_max):
        super().__init__()
        self.scale_mapping_min = scale_mapping_min
        self.scale_mapping_max = scale_mapping_max

    def forward(self, scale):
        # add a small offset to avoid zero point: [0.1 to 1.1]
        scale = (scale - self.scale_mapping_min) / (self.scale_mapping_max - self.scale_mapping_min)
        return scale + 0.1
    
from torch import nn
@persistence.persistent_class
class ScalarEncoder1d(nn.Module):
    """
    1-dimensional Fourier Features encoder (i.e. encodes raw scalars)
    Assumes that scalars are in [0, 1]
    """
    def __init__(self, coord_dim: int, x_multiplier: float, const_emb_dim: int, use_raw: bool=False, **fourier_enc_kwargs):
        super().__init__()
        self.coord_dim = coord_dim
        self.const_emb_dim = const_emb_dim
        self.x_multiplier = x_multiplier
        self.use_raw = use_raw

        if self.const_emb_dim > 0 and self.x_multiplier > 0:
            self.const_embed = nn.Embedding(int(np.ceil(x_multiplier)) + 1, self.const_emb_dim)
        else:
            self.const_embed = None

        if self.x_multiplier > 0:
            self.fourier_encoder = FourierEncoder1d(coord_dim, max_x_value=x_multiplier, **fourier_enc_kwargs)
            self.fourier_dim = self.fourier_encoder.get_dim()
        else:
            self.fourier_encoder = None
            self.fourier_dim = 0

        self.raw_dim = 1 if self.use_raw else 0

    def get_dim(self) -> int:
        return self.coord_dim * (self.const_emb_dim + self.fourier_dim + self.raw_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assumes that x is in [0, 1] range
        """
        misc.assert_shape(x, [None, self.coord_dim])
        batch_size, coord_dim = x.shape
        out = torch.empty(batch_size, self.coord_dim, 0, device=x.device, dtype=x.dtype) # [batch_size, coord_dim, 0]
        if self.use_raw:
            out = torch.cat([out, x.unsqueeze(2)], dim=2) # [batch_size, coord_dim, 1]
        if not self.fourier_encoder is None or not self.const_embed is None:
            # Convert from [0, 1] to the [0, `x_multiplier`] range
            x = x.float() * self.x_multiplier # [batch_size, coord_dim]
        if not self.fourier_encoder is None:
            fourier_embs = self.fourier_encoder(x) # [batch_size, coord_dim, fourier_dim]
            out = torch.cat([out, fourier_embs], dim=2) # [batch_size, coord_dim, raw_dim + fourier_dim]
        if not self.const_embed is None:
            const_embs = self.const_embed(x.round().long()) # [batch_size, coord_dim, const_emb_dim]
            out = torch.cat([out, const_embs], dim=2) # [batch_size, coord_dim, raw_dim + fourier_dim + const_emb_dim]
        out = out.view(batch_size, coord_dim * (self.raw_dim + self.const_emb_dim + self.fourier_dim)) # [batch_size, coord_dim * (raw_dim + const_emb_dim + fourier_dim)]
        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class FourierEncoder1d(nn.Module):
    def __init__(self,
            coord_dim: int,               # Number of scalars to encode for each sample
            max_x_value: float=100.0,       # Maximum scalar value (influences the amount of fourier features we use)
            transformer_pe: bool=False,     # Whether we should use positional embeddings from Transformer
            use_cos: bool=True,
            **construct_freqs_kwargs,
        ):
        super().__init__()
        assert coord_dim >= 1, f"Wrong coord_dim: {coord_dim}"
        self.coord_dim = coord_dim
        self.use_cos = use_cos
        if transformer_pe:
            d_model = 512
            fourier_coefs = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)) # [d_model]
        else:
            fourier_coefs = construct_log_spaced_freqs(max_x_value, **construct_freqs_kwargs)
        self.register_buffer('fourier_coefs', fourier_coefs) # [num_fourier_feats]
        self.fourier_dim = self.fourier_coefs.shape[0]

    def get_dim(self) -> int:
        return self.fourier_dim * (2 if self.use_cos else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, f"Wrong shape: {x.shape}"
        assert x.shape[1] == self.coord_dim
        fourier_raw_embs = self.fourier_coefs.view(1, 1, self.fourier_dim) * x.float().unsqueeze(2) # [batch_size, coord_dim, fourier_dim]
        if self.use_cos:
            fourier_embs = torch.cat([fourier_raw_embs.sin(), fourier_raw_embs.cos()], dim=2) # [batch_size, coord_dim, 2 * fourier_dim]
        else:
            fourier_embs = fourier_raw_embs.sin() # [batch_size, coord_dim, fourier_dim]
        return fourier_embs

#----------------------------------------------------------------------------
from typing import Tuple
def construct_log_spaced_freqs(max_t: int, skip_small_t_freqs: int=0, skip_large_t_freqs: int=0) -> Tuple[int, torch.Tensor]:
    time_resolution = 2 ** np.ceil(np.log2(max_t))
    num_fourier_feats = np.ceil(np.log2(time_resolution)).astype(int)
    powers = torch.tensor([2]).repeat(num_fourier_feats).pow(torch.arange(num_fourier_feats)) # [num_fourier_feats]
    powers = powers[skip_large_t_freqs:len(powers) - skip_small_t_freqs] # [num_fourier_feats]
    fourier_coefs = powers.float() * np.pi # [1, num_fourier_feats]

    return fourier_coefs / time_resolution