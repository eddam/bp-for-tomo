import numpy as np
from util import generate_synthetic_data

def image_generator(n_pts=25, seed=None, save_file=None):
    """
    Generate blob-like binary data and (optionally) save the
    data to a text file.

    Parameters
    ----------

    n_pts: int, default 25
        number of seeds used to generate the structures. The larger n_pts,
        the finer will be the structures.

    seed: int, default 0
        seed with which to initialize the random number generator.

    save_file: bool or str, default None
        if save_file=True, the data are saved in the file binary_data.txt
        if a string is given, the string is used as a filename

    Returns
    -------

    im: ndarray
        the generated image

    Examples
    --------
    >>> image_generator(n_pts=25, save_file=True) # save in binary_data.txt
    >>> image_generator(n_pts=25, save_file='data.txt')
    >>> # Finer structures
    >>> image_generator(n_pts=49) 
    """
    l_x = 256
    im = generate_synthetic_data(l_x, seed=seed, n_pts=n_pts)
    if save_file is not None:
        if save_file == True:
            np.savetxt('binary_data.txt', im.ravel())
        else:
            np.savetxt(save_file, im.ravel())
    return im
