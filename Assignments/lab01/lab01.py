# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import sys

# Allow helpers package to be found when this module is imported standalone
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from helpers.dataloader import load_image, get_data_path

SPHINX_IMAGE = "2560px-Great_Sphinx_of_Giza_-_20080716a.jpg"

def read_image():
    """Load and preprocess the Sphinx image: scale by 0.5 and convert to greyscale."""
    image_path = get_data_path(SPHINX_IMAGE)
    return load_image(image_path, scale_factor=2, as_gray=True)

# Skeleton Code will generate a 1D Gaussian Kernel for given sigma.
# You need to implement 2D convolution as efficiently as possible.
class GaussianFilt:
    def __init__(self, sigma):
        self.sigma = sigma

    def gauss_kernel(self):
        """Generate a normalised 1D Gaussian kernel of appropriate size for self.sigma.

        Returns:
            gauss_1d (ndarray): Shape (k_size, 1) — column vector kernel.
        """
        # Work out the necessary kernel size and range to generate the Gaussian over.
        # raise NotImplementedError("Implement this method")
        sigma = self.sigma
        if sigma <= 0:
            raise ValueError("sigma must be greater than 0")

        k_size = int(6 * sigma + 1)
        if k_size % 2 == 0:
            k_size += 1

        x = np.arange(-(k_size //2), k_size // 2 +1, dtype=np.float64)
        kernel = np.exp(-(x**2) / (2*sigma**2))
        kernel = kernel / kernel.sum()

        return kernel[:, np.newaxis].astype(np.float64)

    def my_conv_method(self, image):
        """Perform separable 2D Gaussian convolution via two 1D matrix-multiply passes.

        Args:
            image (ndarray): 2D greyscale image (H, W), any numeric dtype.

        Returns:
            conv_img (ndarray): Blurred image of same shape, dtype uint8.
        """
        # raise NotImplementedError("Implement this method")
        image = np.asarray(image, dtype=np.float64)

        kernel_col = self.gauss_kernel()
        kernel = kernel_col[:, 0]

        pad = len(kernel) // 2

        padded = np.pad(image, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)

        temp=np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="valid"),
            axis=0,
            arr=padded
        )

        result = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="valid"),
            axis=1,
            arr=temp
        )

        result = np.rint(result)
        result = np.clip(result, 0, 255)

        return result.astype(np.uint8)
