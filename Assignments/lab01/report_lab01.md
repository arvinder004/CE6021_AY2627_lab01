# Lab Report — Lab 01: Efficient Separable Gaussian Convolution

*Fill in each section below. Be specific — name the actual methods/parameters you used, since
your report is graded alongside your code.*

## Overview

_1 sentence: What was the goal of this lab?_

answer - the goal of this lab was to implement gaussian convolution efficiently by exploiting the seperability of the 2D gaussian kernel, replacing a single 2D convolution with 2 1D passes while maintaining correctness and meeting performance benchmarks

## Implementation

_1-2 sentences: Briefly describe the steps you performed to implment efficient convolution._

answer - i implemented the GaussianFilt class by creating the gauss_kernel() method to generate a normalized 1D gaussian kernel of appropriate size for self.sigma returning a column vector kernel of shape (k_size, 1). then i implemented my_conv_method(image) which pads the image once performs 1D convolution along one axis and then performs the same operation along the other axis to produce a blurred image of the same shape as the input converted to np.uint8

## Results

_1-2 sentences: How well did your method perform vs the baseline in the pytests? ._

answer - the implementation was benchmarked against a naive baseline using direct 2D convolve2d with the full gaussian kernel. the tests verified both correctness and execution time and the method passed all checks with runtime substantially lower than the baseline demonstrating that the seperable approach is both accurate and much more efficient.
