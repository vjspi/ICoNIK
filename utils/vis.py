

import matplotlib
import numpy as np
import torch
from medutils.visualization import center_crop
from utils.mri import coilcombine, ifft2c_mri

def k2img(k, csm=None, norm_factor=1.5, scale = True):
    """
    Convert k-space data to image space and generate visual representations.

    Parameters:
        k (torch.Tensor or numpy.ndarray): k-space data on a Cartesian grid. The dimensions can be either 4 or 5.
        csm (torch.Tensor or None, optional): Coil sensitivity maps. If provided, coil combination will be performed using the maps.
        norm_factor (float, optional): Normalization factor for image scaling (default is 1.5).
        scale (bool, optional): Whether to scale the output images to the range [0, 255] (default is True).

    Returns:
        dict: A dictionary containing different visual representations of the data:
            - 'k_mag': Magnitude of the k-space data.
            - 'combined_mag': Combined magnitude image after coil combination.
            - 'combined_phase': Combined phase image after coil combination.
            - 'combined_img': Combined complex image after coil combination.
    """

    if k.ndim == 4:
        coil_img = ifft2c_mri(k)

    elif k.ndim == 5:
        coil_img = torch.empty_like(k)
        for i in range(k.shape[2]):
            coil_img[:, :, i, ...] = ifft2c_mri(k[:, :, i, ...])
        # combined_img_motion = coil_img_motion.abs()

    k_mag = k[:, 0, ...].abs().unsqueeze(1).detach().cpu().numpy()  # nt, nx, ny
    if csm is not None:
        im_shape = csm.shape[-2:]  # (nx, ny)
        combined_img = coilcombine(coil_img, im_shape, coil_dim=1, csm=csm)
    else:
        combined_img = coilcombine(coil_img, coil_dim=1, mode='rss')
    combined_phase = torch.angle(combined_img).detach().cpu().numpy()
    combined_mag = combined_img.abs().detach().cpu().numpy()
    k_mag = np.log(np.abs(k_mag) + 1e-4)

    if scale:
        k_min = np.min(k_mag)
        k_max = np.max(k_mag)
        max_int = 255
        combined_mag_max = combined_mag.max() / norm_factor

        k_mag = (k_mag - k_min) * (max_int) / (k_max - k_min)
        k_mag = np.minimum(max_int, np.maximum(0.0, k_mag))
        k_mag = k_mag.astype(np.uint8)
        combined_mag = (combined_mag / combined_mag_max * 255)  # .astype(np.uint8)
        combined_phase = angle2color(combined_phase, cmap='viridis', vmin=-np.pi, vmax=np.pi)
        k_mag = np.clip(k_mag, 0, 255).astype(np.uint8)
        combined_mag = np.clip(combined_mag, 0, 255).astype(np.uint8)
        combined_phase = np.clip(combined_phase, 0, 255).astype(np.uint8)

    combined_img = combined_img.detach().cpu().numpy()
    vis_dic = {
        'k_mag': k_mag,
        'combined_mag': combined_mag,
        'combined_phase': combined_phase,
        'combined_img': combined_img
    }
    return vis_dic



def alpha2img(alpha, csm=None):
    """
    Convert phase data to images suitable for visualization.

    Parameters:
        alpha (torch.Tensor): Phase data (alpha).
        csm (torch.Tensor or None, optional): Coil sensitivity maps. If provided, coil combination will be performed using the maps.

    Returns:
        dict: A dictionary containing visual representations of the phase data:
            - 'alpha': Phase data converted to an image in [0, 255] range.
            - 'alpha_color': Phase data converted to a color image.
    """

    alpha_img = alpha.detach().cpu().numpy()

    alpha_color = angle2color(alpha_img)  # vmin=-1, vmax=1)

    max_int = 255
    alpha_img = (alpha_img - alpha_img.min()) * max_int / (alpha_img.max() - alpha_img.min())
    alpha_img = np.minimum(max_int, np.maximum(0.0, alpha_img))
    alpha_img = alpha_img.astype(np.uint8)

    alpha_vis = {
        'alpha': alpha_img,
        'alpha_color': alpha_color}
    return alpha_vis

def angle2color(value_arr, cmap='viridis', vmin=None, vmax=None):
    """
    Convert a value to a color using a colormap.

    Parameters:
        value: The value to convert.
        cmap: The colormap to use.

    Returns:
        The color corresponding to the value.
    """
    if vmin is None:
        vmin = value_arr.min()
    if vmax is None:
        vmax = value_arr.max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    try:
        value_arr = value_arr.squeeze(0)
    except:
        value_arr = value_arr.squeeze()
    if len(value_arr.shape) == 3:
        color_arr = np.zeros((*value_arr.shape, 4))
        for i in range(value_arr.shape[0]):
            color_arr[i] = mapper.to_rgba(value_arr[i], bytes=True)
        color_arr = color_arr.transpose(0, 3, 1, 2)
    elif len(value_arr.shape) == 2:
        color_arr = mapper.to_rgba(value_arr, bytes=True)
    return color_arr
