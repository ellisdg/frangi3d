import numpy as np


def divide_nonzero(array1, array2):
    """
    Divides two arrays. Returns zero when dividing by zero.
    """
    denominator = np.copy(array2)
    denominator[denominator == 0] = 1e-10
    return np.divide(array1, denominator)


def create_image_like(data, image):
    return image.__class__(data, affine=image.affine, header=image.header)


def absolute_eigenvaluesh(nd_array):
    """
    Computes the eigenvalues sorted by absolute value from the symmetrical matrix.
    :param nd_array: array from which the eigenvalues will be calculated.
    :return: A list with the eigenvalues sorted in absolute ascending order (e.g. [eigenvalue1, eigenvalue2, ...])
    """
    eigenvalues = np.linalg.eigvalsh(nd_array)
    sorted_eigenvalues = sortbyabs(eigenvalues, axis=-1)
    return [np.squeeze(eigenvalue, axis=-1)
            for eigenvalue in np.split(sorted_eigenvalues, sorted_eigenvalues.shape[-1], axis=-1)]


def sortbyabs(a, axis=0):
    """Sort array along a given axis by the absolute value
    modified from: http://stackoverflow.com/a/11253931/4067734
    """
    index = list(np.ix_(*[np.arange(i) for i in a.shape]))
    index[axis] = np.abs(a).argsort(axis)
    return a[index]
