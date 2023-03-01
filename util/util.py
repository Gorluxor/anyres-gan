import torch
import urllib

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1-decay)

def set_requires_grad(requires_grad, *models):
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, 'unknown type %r' % type(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever

import torch
from torchvision.transforms import ToTensor, Grayscale
import torch.nn.functional as F
from PIL import Image
def grad_img(img, kernel):
    # check if the img is PIL image, if it is convert it to tensor
    if isinstance(img, Image.Image):
        img = ToTensor()(img)
    # check if it has correct batch size, if not add it
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    # check if the kernel is torch tensor, if it is convert it to tensor
    if not isinstance(kernel, torch.Tensor):
        kernel = torch.tensor(kernel)
    # check if the kernel is 2D, if it is convert it to 4D
    if len(kernel.shape) == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0)
    # check if the kernel is 3D, if it is convert it to 4D
    if len(kernel.shape) == 3:
        kernel = kernel.unsqueeze(0)
    # if the image has 3 channels, convert to grayscale
    if img.shape[1] == 3:
        img = Grayscale()(img)
    # perform convolution
    same_padding_size = int((kernel.shape[2] - 1) // 2)
    pad = torch.nn.ReflectionPad2d(same_padding_size)
    img_grad = F.conv2d(pad(img), kernel)
    # sanity check, same dimensions before and after
    assert img_grad.shape == img.shape
    return img_grad

def grad_with_kernel(img:torch.Tensor, kernel:str) -> torch.Tensor:
    """ Use for computing the gradient of an image with a given kernel

    Args:
        img (torch.Tensor): image, shape (B, C, H, W) or (C, H, W)
        kernel (str): kernel type, one of ['tx', 'ty', 'txy']

    Returns:
        torch.Tensor: gradient image, shape (B, C, H, W) or if (C, H, W) will return (1, C, H, W)
    """    
    assert kernel in ['tx', 'ty', 'txy']
    if kernel == 'tx':
        kernel = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32)
    elif kernel == 'ty':
        kernel = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float32)
    elif kernel == 'txy':
        kernel = torch.tensor([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=torch.float32) # diagonal gradient txy
    device = img.device if hasattr(img, 'device') else torch.device('cpu')
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return grad_img(img, kernel.to(device))

def grad_with_all_kernels(img:torch.Tensor, normalize:bool = False) -> torch.Tensor:
    rez = None
    for kernel in ['tx', 'ty', 'txy']:
        if rez is None:
            rez = grad_with_kernel(img.clone(), kernel) 
            # rez = rez.unsqueeze(0) if len(rez.shape) == 3 else rez # returns already good shape
        else:
            rez = torch.cat([rez, grad_with_kernel(img.clone(), kernel)], dim=1)
    # normalize between -1 and 1
    if normalize:
        rez = rez / torch.max(torch.abs(rez))
    return rez

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
