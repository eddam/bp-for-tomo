import numpy as np
from scipy import ndimage


def generate_synthetic_data(l_x=128, seed=None, crop=True, n_pts=25):
    """
    Generate synthetic binary data looking like phase separation

    Parameters
    ----------

    l_x: int, default 128
        Linear size of the returned image

    seed: int, default 0
        seed with which to initialize the random number generator.

    crop: bool, default True
        If True, non-zero data are found only within a central circle
        of radius l_x / 2

    n_pts: int, default 25
        number of seeds used to generate the structures. The larger n_pts,
        the finer will be the structures.

    Returns
    -------

    res: ndarray of float32, of shape lxl
        Output binary image

    Examples
    --------
    >>> im = generate_synthetic_data(l_x=256, seed=2, n_pts=25)
    >>> # Finer structures
    >>> im = generate_synthetic_data(l_x=256, n_pts=100)
    """
    if seed is None:
        seed = 0
    # Fix the seed for reproducible results
    rs = np.random.RandomState(seed)
    x, y = np.ogrid[:l_x, :l_x]
    mask = np.zeros((l_x, l_x))
    points = l_x * rs.rand(2, n_pts)
    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    mask = ndimage.gaussian_filter(mask, sigma=l_x / (4. * np.sqrt(n_pts)))
    # Limit the non-zero data to a central circle
    if crop:
        mask_outer = (x - l_x / 2) ** 2 + (y - l_x / 2) ** 2 < (l_x / 2) ** 2
        mask = np.logical_and(mask > mask.mean(), mask_outer)
    else:
        mask = mask > mask.mean()
    return mask.astype(np.float32)

def compute_best_J(im):
    # A more precise computation of sisj could be done...
    l_x = len(im)
    X, Y = np.ogrid[:l_x, :l_x]
    mask = ((X - l_x/2)**2 + (Y - l_x/2)**2 <= (l_x/2)**2)
    grad = ndimage.morphological_gradient(im, footprint=np.ones((3, 3)))
    sisj_average = 1 - (grad[mask] > 0).mean()
    J1 = np.arctanh(sisj_average)
    grad2 = np.abs(np.diff(im, axis=0))[:, :-1] + np.abs(np.diff(im, axis=1))[:-1]
    sisj_average = 1 - 2*(grad2[mask[:-1, :-1]] > 0).mean()
    J2 = np.arctanh(sisj_average)
    return J1, J2

def compute_sparsity(im):
    l_x = len(im)
    X, Y = np.ogrid[:l_x, :l_x]
    mask = ((X - l_x/2)**2 + (Y - l_x/2)**2 <= (l_x/2)**2)
    grad1 = ndimage.morphological_gradient(im, footprint=np.ones((3, 3)))
    grad2 = ndimage.morphological_gradient(im, footprint=ndimage.generate_binary_structure(2, 1))
    return (grad1[mask] > 0).mean(), (grad2[mask] > 0).mean() 
