import numpy as np
import math
import torch
import torch.nn.functional as F
import typing


def get_gausskernel_size(sigma, force_odd=True):
    ksize = 2 * math.ceil(sigma * 3.0) + 1
    if ksize % 2 == 0 and force_odd:
        ksize += 1
    return int(ksize)


def gaussian1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Function that computes values of a (1D) Gaussian with zero mean and variance sigma^2
    """
    out = -x ** 2
    out /= 2 * sigma ** 2
    out = torch.div(torch.exp(out), np.sqrt(2 * np.pi) * sigma)
    return out


def gaussian_deriv1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Function that computes values of a (1D) Gaussian derivative
    """
    return -x / (sigma ** 2) * gaussian1d(x, sigma)


def filter2d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Function that convolves a tensor with a kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.
    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(kH, kW)`.
    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    hout = (kernel.shape[0] - 1) / 2
    wout = (kernel.shape[1] - 1) / 2
    pad = tuple(np.array([wout, wout, hout, hout]).astype(int))

    x = torch.nn.functional.pad(x, pad, mode="replicate")

    out = torch.nn.functional.conv2d(x,
                                     kernel.unsqueeze(0).repeat(x.shape[1], 1, 1, 1),
                                     groups=x.shape[1])
    return out


def gaussian_filter2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Function that blurs a tensor using a Gaussian filter.

    Arguments:
        sigma (Tuple[float, float]): the standard deviation of the kernel.
        
    Returns:
        Tensor: the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`
    """
    ksize = get_gausskernel_size(sigma)
    inp = torch.linspace(-ksize // 2 + 1, ksize // 2, ksize)
    kernel = gaussian1d(inp, sigma).unsqueeze(0)
    out = filter2d(x, kernel)
    return filter2d(out, kernel.T)


def spatial_gradient_first_order(x: torch.Tensor,
                                 sigma: float) -> torch.Tensor:
    """
    Computes the first order image derivative in both x and y directions using Gaussian derivative

    Return:
        torch.Tensor: spatial gradients

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`
    """
    b, c, h, w = x.shape
    kernel = torch.tensor([[.5, 0.0, -.5]])

    xy = filter2d(gaussian_filter2d(x, sigma), kernel).unsqueeze(2)
    yx = filter2d(gaussian_filter2d(x, sigma), kernel.T).unsqueeze(2)
    return torch.cat([xy, yx], dim=2)


def spatial_gradient_second_order(x: torch.Tensor,
                                  sigma: float) -> torch.Tensor:
    """
    Computes the second order image derivative in xx, xy, yy directions using Gaussian derivative

    Return:
        torch.Tensor: spatial gradients

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 3, H, W)`

    """
    b, c, h, w = x.shape

    kernel = torch.tensor([[.5, 0.0, -.5]])

    fo = spatial_gradient_first_order(x, sigma)
    dx, dy = fo[:, :, 0], fo[:, :, 1]

    xx = filter2d(dx, kernel).unsqueeze(2)
    yy = filter2d(dy, kernel.T).unsqueeze(2)
    xy = filter2d(dx, kernel.T).unsqueeze(2)

    # return torch.cat([xx, xy, yy], dim=2)
    return spatial_gradient_second_order_fast(x, sigma)


# Reference solution for hw0
def spatial_gradient_first_order_fast(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Computes the first order image derivative in both x and y directions using Gaussian derivative

    Return:
        torch.Tensor: spatial gradients

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`

    """
    b, c, h, w = x.shape
    ksize = get_gausskernel_size(sigma)
    # HIDE_IN_TEMPLATE_START
    kernel_inp = torch.linspace(-float(ksize // 2), float(ksize // 2), ksize)
    filtered_input = gaussian_filter2d(x, sigma)
    gx = torch.tensor([[0.5, 0, -0.5]]).float()  # .view(1,-1)
    # g = gaussian1d(kernel_inp, sigma).reshape(1,-1)
    # kernel_x = torch.mm(g.t(),gx)
    outx = filter2d(filtered_input, gx)
    outy = filter2d(filtered_input, gx.t())
    out = torch.cat([outx.unsqueeze(2), outy.unsqueeze(2)], dim=2)
    # HIDE_IN_TEMPLATE_STOP
    # SHOW_IN_TEMPLATE out =  torch.zeros(b,c,2,h,w)
    return out


# Reference solution for hw0
def filter2d_fast(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Function that convolves a tensor with a kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(kH, kW)`.
    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    # SHOW_IN_TEMPLATE out =  x
    # HIDE_IN_TEMPLATE_START
    b, c, h, w = x.shape
    height, width = kernel.size()
    tmp_kernel: torch.Tensor = kernel[None, None, ...].to(x.device).to(x.dtype)
    padding_shape = [width // 2, width // 2, height // 2, height // 2]
    input_pad: torch.Tensor = F.pad(x, padding_shape, mode='replicate')
    out = F.conv2d(input_pad, tmp_kernel.expand(c, -1, -1, -1), groups=c, padding=0, stride=1)
    # HIDE_IN_TEMPLATE_STOP
    # SHOW_IN_TEMPLATE out =  x
    return out


# Reference solution for hw0
def spatial_gradient_second_order_fast(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Computes the second order image derivative in xx, xy, yy directions using Gaussian derivative

    Return:
        torch.Tensor: spatial gradients

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 3, H, W)`

    """
    b, c, h, w = x.shape
    ksize = get_gausskernel_size(sigma)
    # HIDE_IN_TEMPLATE_START
    gx = torch.tensor([[0.5, 0, -0.5]]).to(x.device, x.dtype)
    first_order = spatial_gradient_first_order_fast(x, sigma)
    out = torch.cat([filter2d_fast(first_order[:, :, 0], gx).unsqueeze(2),
                     filter2d_fast(first_order[:, :, 0], gx.t()).unsqueeze(2),
                     filter2d_fast(first_order[:, :, 1], gx.t()).unsqueeze(2)], dim=2)

    # HIDE_IN_TEMPLATE_STOP
    # SHOW_IN_TEMPLATE out =  torch.zeros(b,c,3,h,w)
    return out


def affine(center: torch.Tensor, unitx: torch.Tensor,
           unity: torch.Tensor) -> torch.Tensor:
    """
    Computes transformation matrix A which transforms point in homogeneous
    coordinates from canonical coordinate system into image

    Return:
        torch.Tensor: affine transformation matrix

    Shape:
        - Input :math:`(B, 2)`, :math:`(B, 2)`, :math:`(B, 2)` 
        - Output: :math:`(B, 3, 3)`

    """
    assert center.size(0) == unitx.size(0)
    assert center.size(0) == unity.size(0)
    B = center.size(0)
    out = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    out[:, 0, 0] = unitx[:, 0] - center[:, 0]
    out[:, 0, 1] = unity[:, 0] - center[:, 0]
    out[:, 0, 2] = center[:, 0]

    out[:, 1, 0] = unitx[:, 1] - center[:, 1]
    out[:, 1, 1] = unity[:, 1] - center[:, 1]
    out[:, 1, 2] = center[:, 1]
    return out


def extract_affine_patches(imgs: torch.Tensor,
                           A: torch.Tensor,
                           img_idxs: torch.Tensor,
                           PS: int = 32,
                           ext: float = 6.0):
    """
    Extract patches defined by affine transformations A from image tensor X.
    
    Args:
        imgs: (torch.Tensor) images, :math:`(B, CH, H, W)`
        A: (torch.Tensor). :math:`(N, 3, 3)`
        img_idxs: (torch.Tensor). :math:`(N, 1)` indexes of image in batch, where patch belongs to
        PS: (int) output patch size in pixels, default = 32
        ext (float): output patch size in unit vectors. 

    Returns:
        patches: (torch.Tensor) :math:`(N, CH, PS,PS)`
    """

    b, ch, h, w = imgs.size()
    num_patches = A.size(0)

    images = imgs[img_idxs]

    # Functions, which might be useful: torch.meshgrid, torch.nn.functional.grid_sample
    # You are not allowed to use function torch.nn.functional.affine_grid
    # Note, that F.grid_sample expects coordinates in a range from -1 to 1
    # where (-1, -1) - topleft, (1,1) - bottomright and (0,0) center of the image
    # with torch.no_grad():
    #     grid_x, grid_y = torch.meshgrid(torch.linspace(-ext, ext, PS),
    #                                     torch.linspace(-ext, ext, PS))
    #     grid_x, grid_y = grid_x.repeat(num_patches, 1, 1), grid_y.repeat(num_patches, 1, 1)
    #     grid_x, grid_y = grid_x.flatten(start_dim=1), grid_y.flatten(start_dim=1)
    #
    #     # to homogeneous coordinates, to image coordinates
    #     grid = torch.cat((grid_x.unsqueeze(1), grid_y.unsqueeze(1), torch.ones(grid_y.size()).unsqueeze(1)), dim=1)
    #     grid = torch.cat([(A[i] @ grid[i])[:-1].unsqueeze(0) for i in range(num_patches)], dim=0)
    #
    #     # resize to (N, H_p, W_p, 2)
    #     grid = torch.cat([torch.cat([grid[:, c, i * PS:(i + 1) * PS].unsqueeze(2) for i in range(PS)], dim=2).unsqueeze(3) for c in range(2)], dim=3)
    #
    #     _, _, h, w = imgs.size()
    #     grid[..., 0] = 2 * grid[..., 0] / w - 1  # coordinated to [-1, 1] x [-1, 1] range
    #     grid[..., 1] = 2 * grid[..., 1] / h - 1
    #
    #     out = F.grid_sample(imgs[img_idxs], grid, mode='bilinear')
    # return out

    inp = torch.linspace(-ext, ext, PS)

    gridy, gridx = torch.meshgrid(inp, inp)
    gridz = torch.ones_like(gridx)

    init_grid = torch.zeros(num_patches, 3, inp.shape[0], inp.shape[0])

    init_grid[:, 0] = gridx
    init_grid[:, 1] = gridy
    init_grid[:, 2] = gridz
    right_mat = torch.reshape(init_grid, [num_patches, 3, inp.shape[0] ** 2])
    res_grid = A @ right_mat
    res_grid = torch.reshape(res_grid, [num_patches, 3, inp.shape[0], inp.shape[0]])[:, :2]
    res_grid[:, 0] = (res_grid[:, 0] * 2 / w) - 1
    res_grid[:, 1] = (res_grid[:, 1] * 2 / h) - 1

    res_grid = torch.stack((res_grid[:, 0], res_grid[:, 1]), 3)
    out = torch.nn.functional.grid_sample(images, res_grid, mode='bilinear')

    return out


def extract_antializased_affine_patches(inp: torch.Tensor,
                                        A: torch.Tensor,
                                        img_idxs: torch.Tensor,
                                        PS: int = 32,
                                        ext: float = 6.0):
    """
    Extract patches defined by affine transformations A from scale pyramid created image tensor X.
    It runs your implementation of the `extract_affine_patches` function, so it would not work w/o it.
    
    Args:
        inp: (torch.Tensor) images, :math:`(B, CH, H, W)`
        A: (torch.Tensor). :math:`(N, 3, 3)`
        img_idxs: (torch.Tensor). :math:`(N, 1)` indexes of image in batch, where patch belongs to
        PS: (int) output patch size in pixels, default = 32
        ext (float): output patch size in unit vectors. 

    Returns:
        patches: (torch.Tensor) :math:`(N, CH, PS,PS)`
    """
    import kornia
    b, ch, h, w = inp.size()
    num_patches = A.size(0)
    scale = (kornia.feature.get_laf_scale(
            ext * A.unsqueeze(0)[:, :, :2, :]) / float(PS))[0]
    half: float = 0.5
    pyr_idx = (scale.log2()).relu().long()
    cur_img = inp
    cur_pyr_level = 0
    out = torch.zeros(num_patches, ch, PS, PS).to(device=A.device,
                                                  dtype=A.dtype)
    while min(cur_img.size(2), cur_img.size(3)) >= PS:
        _, ch_cur, h_cur, w_cur = cur_img.size()
        scale_mask = (pyr_idx == cur_pyr_level).squeeze()
        if (scale_mask.float().sum()) >= 0:
            scale_mask = (scale_mask > 0).view(-1)
            current_A = A[scale_mask]
            current_A[:, :2, :3] *= (float(h_cur) / float(h))
            patches = extract_affine_patches(cur_img,
                                             current_A,
                                             img_idxs[scale_mask],
                                             PS, ext)
            out.masked_scatter_(scale_mask.view(-1, 1, 1, 1), patches)
        cur_img = kornia.geometry.pyrdown(cur_img)
        cur_pyr_level += 1
    return out
