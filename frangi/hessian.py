from itertools import combinations_with_replacement

import numpy as np
from scipy import ndimage as ndi

from .utils import absolute_eigenvaluesh


def compute_hessian_matrix(nd_array, sigma=1, scale=True):
    """
    Computes the hessian matrix for an nd_array.
    This can be used to detect vesselness as well as other features.

    In 3D the first derivative will contain three directional gradients at each index:
    [ gx,  gy,  gz ]

    The Hessian matrix at each index will then be equal to the second derivative:
    [ gxx, gxy, gxz]
    [ gyx, gyy, gyz]
    [ gzx, gzy, gzz]

    The Hessian matrix is symmetrical, so gyx == gxy, gzx == gxz, and gyz == gzy.

    :param nd_array: n-dimensional array from which to compute the hessian matrix.
    :param sigma: gaussian smoothing to perform on the array.
    :param scale: if True, the hessian elements will be scaled by sigma squared.
    :return: hessian array of shape (..., ndim, ndim)
    """
    ndim = nd_array.ndim

    # smooth the nd_array
    smoothed = ndi.gaussian_filter(nd_array, sigma=sigma)

    # compute the first order gradients
    gradient_list = np.gradient(smoothed)

    # compute the hessian elements
    hessian_elements = [np.gradient(gradient_list[ax0], axis=ax1)
                        for ax0, ax1 in combinations_with_replacement(range(ndim), 2)]

    if sigma > 0 and scale:
        # scale the elements of the hessian matrix
        hessian_elements = [(sigma ** 2) * element for element in hessian_elements]

    # create hessian matrix from hessian elements
    hessian_full = [[None] * ndim] * ndim

    for index, (ax0, ax1) in enumerate(combinations_with_replacement(range(ndim), 2)):
        element = hessian_elements[index]
        hessian_full[ax0][ax1] = element
        if ax0 != ax1:
            hessian_full[ax1][ax0] = element

    hessian_rows = list()
    for row in hessian_full:
        hessian_rows.append(np.stack(row, axis=-1))

    hessian = np.stack(hessian_rows, axis=-2)
    return hessian


def absolute_hessian_eigenvalues(nd_array, sigma=1, scale=True):
    """
    Eigenvalues of the hessian matrix calculated from the input array sorted by absolute value.
    :param nd_array: input array from which to calculate hessian eigenvalues.
    :param sigma: gaussian smoothing parameter.
    :param scale: if True hessian values will be scaled according to sigma squared.
    :return: list of eigenvalues [eigenvalue1, eigenvalue2, ...]
    """
    return absolute_eigenvaluesh(compute_hessian_matrix(nd_array, sigma=sigma, scale=scale))
