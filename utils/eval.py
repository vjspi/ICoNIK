import numpy as np
from medutils.measures import psnr, ssim, nrmse, nrmseAbs
import medutils
import matplotlib.pyplot as plt
import math

def bias_corr(img, ref, mag=True):
    """
    Corrects bias between two complex images.

    Performs bias correction for two complex images using least squares regression.

    Parameters:
        img (numpy.ndarray): The input complex image to be corrected.
        ref (numpy.ndarray): The reference complex image to be used for correction.
        mag (bool, optional): Whether to use magnitude-only for correction (default is True).

    Returns:
        numpy.ndarray: The bias-corrected complex image.

    Note:
        If `mag` is set to True, the magnitude of the input and reference images will be used for correction.
        Otherwise, both the real and imaginary parts will be used.
    """
    if mag:
        ref = np.abs(ref)
        img = np.abs(img)

    im_size = img.shape
    i_flat = img.flatten()
    i_flat = np.concatenate((i_flat.real, i_flat.imag))
    a = np.stack([i_flat, np.ones_like(i_flat)], axis=1)

    b = ref.flatten()
    b = np.concatenate((b.real, b.imag))

    x = np.linalg.lstsq(a, b, rcond=None)[0]
    temp = img.flatten() * x[0] + x[1]
    img_corr = np.reshape(temp, im_size)

    return img_corr


def get_eval_metrics(img, ref):
    """
    Calculate evaluation metrics for an image compared to a reference image.

    Parameters:
        img (numpy.ndarray): The input image to be evaluated.
        ref (numpy.ndarray): The reference image to compare against.

    Returns:
        dict: A dictionary containing evaluation metrics (SSIM, PSNR, NRMSE, and NRMSE with absolute values).
    """
    metrics_dict = {
        "ssim": round(ssim(img, ref), 3),
        "psnr": round(psnr(img, ref), 3),
        "nrmse": round(nrmse(img, ref), 3),
        "nrmseAbs": round(nrmseAbs(img, ref), 3)
    }

    for key, value in metrics_dict.items():
        print(key, ":", value)

    return metrics_dict


def get_eval_metrics_dict(img_dict, ref):
    """
    Calculate evaluation metrics for a dictionary of images compared to a reference image.

    Parameters:
        img_dict (dict): A dictionary containing image names as keys and corresponding images as values.
        ref (numpy.ndarray): The reference image to compare against.

    Returns:
        dict: A dictionary containing evaluation metrics for each image in the input dictionary.
    """
    eval_list = {}

    for key, img in img_dict.items():
        if img is not None:
            metrics_dict = {
                "ssim": round(ssim(img, ref), 3),
                "psnr": round(psnr(img, ref), 3),
                "nrmse": round(nrmse(img, ref), 3),
                "nrmseAbs": round(nrmseAbs(img, ref), 3)
            }
        else:
            metrics_dict = {
                "ssim": None,
                "psnr": None,
                "nrmse": None,
                "nrmseAbs": None
            }

        eval_list[key] = metrics_dict

    return eval_list
