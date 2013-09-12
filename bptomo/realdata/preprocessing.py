import numpy as np
from skimage import util
from bptomo import build_projection_operator
import os
from sklearn.mixture import GMM
from scipy import optimize
from math import exp
from skimage import morphology

def _balance_probs(x, weights, means, sigmasq):
    """
    Difference of probabilty of two Gaussians
    """
    w0, w1 = weights
    m0, m1 = means
    sq0, sq1 = sigmasq
    return w0 * exp(- (x - m0)**2 / (2*sq0)) - w1 * exp(- (x - m1)**2 / (2*sq1))


def extract_mask_and_levels(filename, small_component_size=200):
    """
    Given a first reconstruction, extract the pixels of interest (i.e.,
    pixels of the binary object, as opposed to voids) and the absorption
    levels of the two phases.

    Parameters
    ----------

    filename : str
        Path of the raw data corresponding to one reconstructed slice. It is
        assumed that the data are stored as float32, without any header (as
        slices are reconstruted by the ESRF PyHST software).

    small_component_size : float, default 200
        Minimal size of connected components corresponding to the void phase.
        After the void region is extracted from a statistical model (mixture
        of gaussians), morphological operations are performed to remove small
        connected components corresponding to this phase, to have only one
        large components. Change this value if you have voids inside the
        object of interest, such as bubbles.

    Returns
    -------

    im : np.ndarray
        The reconstructed slice

    inner_mask : np.ndarray of bools
        Mask of the binary object of interest, i.e. pixels that have to be
        determined by BP-tomo

    levels : list with two values
        Absorption values of the two phases, that are used to transform the
        sinogram to correspond to (-1, 1) levels.

    equal_prob : float
        Absorption value that separates the best the two phases. Can be used
        for a first segmentation of the binary object.

    Notes
    -----

    This function used the gaussian mixture classifier from scikit-learn in
    order to separate the image in three (void, phase1, phase2) then two.
    regions
    """
    total_size = os.path.getsize(filename)
    l_x = np.sqrt(total_size / 4)
    im = np.copy(np.memmap(filename, shape=(l_x, l_x), dtype=np.float32))
    X, Y = np.ogrid[:l_x, :l_x]
    mask = (X - l_x/2)**2 + (Y - l_x/2)**2 < (l_x/2)**2
    gmm = GMM(3)
    # Undersampling to save some time
    gmm.fit(im[::2, ::2][mask[::2, ::2]])
    order = np.argsort(gmm.means_.ravel())
    weights = gmm.weights_.ravel()[order]
    means = gmm.means_.ravel()[order]
    sigmasq = gmm.covars_.ravel()[order]
    equal_prob = optimize.brentq(_balance_probs, means[0], means[1],
                        args=(weights[:-1], means[:-1], sigmasq[:-1]))
    inner_mask = im > equal_prob
    inner_clean = morphology.remove_small_objects(inner_mask, 200)
    inner_clean = np.logical_not(morphology.remove_small_objects(
                            np.logical_not(inner_clean), small_component_size))
    gmm = GMM(2)
    gmm.fit(im[::2, ::2][inner_clean[::2, ::2]])
    means = gmm.means_.ravel()
    weights = gmm.weights_.ravel()
    sigmasq = gmm.covars_.ravel()
    equal_prob = optimize.brentq(_balance_probs, means[0], means[1],
                        args=(weights, means, sigmasq))
    return im, inner_clean, gmm.means_.ravel(), equal_prob


def bin_sinogram(sino, bin_detector=2, bin_proj=1):
    """
    Bin the sinogram to reduce the linear size of the detector or the
    number of projections

    Parameters
    ----------

    sino : 2-d np.narray
        Sinogram; each line correspond to a different angle

    bin_detector : int
        Binning parameter along the detector dimension

    bin_proj : int
        Binning parameter along the projections. This will average successive
        projections together, like what is done in a continuous acquisition.
        If you want to undersample the projections but not average them,
        use slicing instead.

    Returns
    -------

    bin_sino : np.ndarray of size
            (sino.shape[0] / bin_proj, sino.shape[1] / bin_detector)
    """
    bin_sino = util.view_as_blocks(sino, block_shape=(bin_proj, bin_detector))
    bin_sino = bin_sino.sum(axis=-1)
    if bin_proj > 1:
        bin_sino = bin_sino.sum(axis=-1)
    return np.squeeze(bin_sino)


def bin_mask(mask, bin_size=2):
    mask = mask.astype(np.float)
    bin_mask = util.view_as_blocks(mask, block_shape=(bin_size, bin_size))
    bin_mask = np.sum(np.sum(bin_mask, axis=-1), axis=-1)
    bin_mask /= bin_size**2
    return bin_mask > 0.5


def rescale_sino_to_binary(sino, v_down, v_up, mask=None, op=None):
    """
    Transform an experimental sinogram obtained from a binary object
    with levels v_down and v_up, to the sinogram obtained from an object
    with the same geometry, but levels -1 and 1.

    Parameters
    ----------

    sino : np.ndarray
        Sinogram; each line correspond to a different angle

    v_down : float
        Smallest value out of the two binary levels

    v_up : float
        Largest value out of the two binary levels

    mask : np.ndarray
        Mask of pixels of interest. Pixels where ``mask`` is False are
        considered to belong to the background (void) and are not
        considered.

    Returns
    -------

    rescaled_sino : np.ndarray
        Transformed sinogram, that can be passed to BP-tomo for binary
        reconstruction.
    """
    # Compute affine rescaling for the absoption values
    if v_up < v_down:
        tmp = v_up
        v_up = v_down
        v_down = tmp
    a = 2. / (v_up - v_down)
    b = 1 - 2. * float(v_up) / (v_up - v_down)
    n_dir, l_x = sino.shape
    if op is None:
        op = build_projection_operator.build_projection_operator(l_x,
                                        n_dir, mask=mask)
    lengths_of_rays = np.array(op.sum(axis=1)).reshape(sino.shape)
    rescaled_sino = a * sino + b * lengths_of_rays
    return rescaled_sino, lengths_of_rays
