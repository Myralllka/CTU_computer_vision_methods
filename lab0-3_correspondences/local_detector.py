import numpy as np
import math
import torch
import torch.nn.functional as F
import typing
from imagefiltering import *


def hessian_response(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Computes the determinant of the Hessian matrix.
    The response map is computed according the following formulation:
    .. math::
        R = det(H)
    where:

    .. math::
        M = \sum_{(x,y) \in W}
        \begin{bmatrix}
            I_{xx} & I_{xy} \\
            I_{xy} & I_{yy} \\
        \end{bmatrix}

    Args:
        x: torch.Tensor: 4d tensor
        sigma (float): sigma of Gaussian derivative

    Return:
        torch.Tensor: Hessian response

    Shape:
       - Input: :math:`(B, C, H, W)`
       - Output: :math:`(B, C, H, W)`
    """
    out_derivatives_sec_ord = spatial_gradient_second_order(x, sigma)
    det_H = out_derivatives_sec_ord[:, :, 0] * out_derivatives_sec_ord[:, :, 2] - \
            out_derivatives_sec_ord[:, :, 1] ** 2
    # out = torch.zeros_like(x)
    return det_H


def harris_response(x: torch.Tensor,
                    sigma_d: float,
                    sigma_i: float,
                    alpha: float = 0.04) -> torch.Tensor:
    """Computes the Harris cornerness function.The response map is computed according the following formulation:

    .. math::
        R = det(M) - alpha \cdot trace(M)^2

    where:

    .. math::
        M = \sum_{(x,y) \in W}
        \begin{bmatrix}
            I^{2}_x & I_x I_y \\
            I_x I_y & I^{2}_y \\
        \end{bmatrix}

    and :math:`k` is an empirically determined constant
    :math:`k ∈ [ 0.04 , 0.06 ]`

    Args:
        x: torch.Tensor: 4d tensor
        sigma_d (float): sigma of Gaussian derivative
        sigma_i (float): sigma of Gaussian blur, aka integration scale
        alpha (float): constant

    Return:
        torch.Tensor: Harris response

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, H, W)`
    """
    out_derivatives_fst_ord = spatial_gradient_first_order(x, sigma_d)

    d2x = out_derivatives_fst_ord[:, :, 0] ** 2
    d2y = out_derivatives_fst_ord[:, :, 1] ** 2
    dxy = out_derivatives_fst_ord[:, :, 0] * out_derivatives_fst_ord[:, :, 1]

    d2x_conv = gaussian_filter2d(d2x, sigma_i)
    d2y_conv = gaussian_filter2d(d2y, sigma_i)
    dxy_conv = gaussian_filter2d(dxy, sigma_i)

    harris = (d2x_conv * d2y_conv - dxy_conv ** 2) - alpha * (d2x_conv + d2y_conv) ** 2
    return harris


def nms2d(x: torch.Tensor, th: float = 0):
    r"""Applies non maxima suppression to the feature map in 3x3 neighborhood.
    Args:
        x: torch.Tensor: 4d tensor
        th (float): threshold
    Return:
        torch.Tensor: nmsed input

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, H, W)`
    """
    b, c, h, w = x.shape
    kernel = torch.eye(9)
    kernel[4, 4] = 0
    kernel = kernel.view(9, 1, 3, 3)
    kernel = kernel.repeat(c, 1, 1, 1)
    
    convolved = F.conv2d(F.pad(x, [1, 1, 1, 1], mode='replicate'), kernel, groups=c)
    convolved = convolved.view(b, c, -1, h, w)
    maximum = convolved.max(dim=2)[0]
    nms_mask = (x > maximum) & (x >= th)
    return x * nms_mask

def hessian(x: torch.Tensor, sigma: float, th: float = 0):
    r"""Returns the coordinates of maximum of the Hessian function.
    Args:
        x: torch.Tensor: 4d tensor
        sigma (float): scale
        th (float): threshold

    Return:
        torch.Tensor: coordinates of local maxima in format (b,c,h,w)

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(N, 4)`, where N - total number of maxima and 4 is (b,c,h,w) coordinates
    """
    # To get coordinates of the responces, you can use torch.nonzero function
    resp = hessian_response(x, sigma)
    return torch.nonzero(nms2d(resp, th))




def harris(x: torch.Tensor, sigma_d: float, sigma_i: float, th: float = 0):
    r"""Returns the coordinates of maximum of the Harris function.
    Args:
        x: torch.Tensor: 4d tensor
        sigma_d (float): scale
        sigma_i (float): scale
        th (float): threshold

    Return:
        torch.Tensor: coordinates of local maxima in format (b,c,h,w)

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(N, 4)`, where N - total number of maxima and 4 is (b,c,h,w) coordinates
    """
    # To get coordinates of the responces, you can use torch.nonzero function
    resp = harris_response(x, sigma_d, sigma_i)
    return torch.nonzero(nms2d(resp, th))


def create_scalespace(x: torch.Tensor, n_levels: int, sigma_step: float):
    r"""Creates an scale pyramid of image, usually used for local feature
    detection. Images are consequently smoothed with Gaussian blur.
    Args:
        x: torch.Tensor :math:`(B, C, H, W)`
        n_levels (int): number of the levels.
        sigma_step (float): blur step.

    Returns:
        Tuple(torch.Tensor, List(float)):
        1st output: image pyramid, (B, C, n_levels, H, W)
        2nd output: sigmas (coefficients for scale conversion)
    """

    b, ch, h, w = x.size()
    
    out = x.unsqueeze(2).repeat(1, 1, n_levels, 1, 1)
    sigmas = [sigma_step ** i for i in range(n_levels)]
    x = torch.cat([gaussian_filter2d(out[:, :, i], sigmas[i]).unsqueeze(2) for i in range(n_levels)], dim=2)
    
    return x, sigmas

    
def nms3d(x: torch.Tensor, th: float = 0):
    r"""Applies non maxima suppression to the scale space feature map in 3x3x3 neighborhood.
    Args:
        x: torch.Tensor: 5d tensor
        th (float): threshold
    Shape:
      - Input: :math:`(B, C, D, H, W)`
      - Output: :math:`(B, C, D, H, W)`
    """
    b, c, d, h, w = x.shape
    
    kernel = torch.eye(27)
    kernel[13, 13] = 0
    kernel = kernel.view(27, 1, 3, 3, 3)
    kernel = kernel.repeat(c, 1, 1, 1, 1)
    
    convolved = F.conv3d(F.pad(x, [1, 1, 1, 1, 1, 1], mode='replicate'), kernel, groups=c, stride=1)
    convolved = convolved.view(b, c, -1, d, h, w)
    
    maximum = convolved.max(dim=2)[0]
    nms_mask = (x > maximum) & (x >= th)
    return x * nms_mask


def scalespace_hessian_response(x: torch.Tensor,
                                n_levels: int = 40,
                                sigma_step: float = 1.1):
    r"""First computes scale space and then computes the determinant of Hessian matrix on 
    Args:
        x: torch.Tensor: 4d tensor
        n_levels (int): number of the levels, (default 40)
        sigma_step (float): blur step, (default 1.1)

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, N_LEVELS, H, W)`, List(floats)
    """
    
    pyr, sigmas = create_scalespace(x, n_levels, sigma_step)
    x = [sigmas[i] ** 4 * hessian_response(x.squeeze(2), 1.9) for x, i in zip(torch.split(pyr,1,2), range(n_levels))]
    resps = torch.stack(x, dim=2)
    return resps, sigmas


def scalespace_hessian(x: torch.Tensor,
                       th: float = 0,
                       n_levels: int = 40,
                       sigma_step: float = 1.1):
    r"""Returns the coordinates of maximum of the Hessian function.
    Args:
        x: torch.Tensor: 4d tensor
        th (float): threshold
        n_levels (int): number of scale space levels (default 40)
        sigma_step (float): blur step, (default 1.1)
        
    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(N, 5)`, where N - total number of maxima and 5 is (b,c,d,h,w) coordinates
    """
    # To get coordinates of the responces, you can use torch.nonzero function
    # Don't forget to convert scale index to scale value with use of sigma
    resps, sigmas = scalespace_hessian_response(x, n_levels, sigma_step)
    
    res = nms3d(resps, th)
    res = torch.nonzero(res)
    sigmas = torch.tensor([sigmas]).view(n_levels)
    res[:, 2] = sigmas[res[:, 2]]
  
    return res

    
def scalespace_harris_response(x: torch.Tensor,
                               n_levels: int = 40,
                               sigma_step: float = 1.1):
    r"""First computes scale space and then computes the Harris cornerness function 
    Args:
        x: torch.Tensor: 4d tensor
        n_levels (int): number of the levels, (default 40)
        sigma_step (float): blur step, (default 1.1)

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, N_LEVELS, H, W)`, List(floats)
    """
    pyr, sigmas = create_scalespace(x, n_levels, sigma_step)
    x = [sigmas[i] ** 4 * harris_response(x.squeeze(2), 0.5, 1.3, 0.04) for x, i in zip(torch.split(pyr,1,2), range(n_levels))]
    resps = torch.stack(x, dim=2)
    return resps, sigmas


def scalespace_harris(x: torch.Tensor,
                      th: float = 0,
                      n_levels: int = 40,
                      sigma_step: float = 1.1):
    r"""Returns the coordinates of maximum of the Harris function.
    Args:
        x: torch.Tensor: 4d tensor
        th (float): threshold
        n_levels (int): number of scale space levels (default 40)
        sigma_step (float): blur step, (default 1.1)
        
    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(N, 5)`, where N - total number of maxima and 5 is (b,c,d,h,w) coordinates
    """
    # To get coordinates of the responces, you can use torch.nonzero function
    # Don't forget to convert scale index to scale value with use of sigma
    resps, sigmas = scalespace_harris_response(x, n_levels, sigma_step)
    
    res = nms3d(resps, th)
    res = torch.nonzero(res)
    sigmas = torch.tensor([sigmas]).view(n_levels)
    res[:, 2] = sigmas[res[:, 2]]
  
    return res
