import numpy as np
from scipy import sparse

def build_projection_operator(l_x, n_dir):
    X, Y = _generate_center_coordinates(l_x)
    angles = np.linspace(0, np.pi, n_dir, endpoint=False)
    data_inds, detector_inds = [], []
    # Indices for data pixels. For each data, one data pixel
    # will contribute to the value of two detector pixels.
    data_unravel_indices = np.arange(l_x ** 2, dtype=np.int32)
    for i, angle in enumerate(angles):
        # rotate data pixels centers
        Xrot = np.cos(angle) * X - np.sin(angle) * Y
        # compute linear interpolation weights
        inds = _weights_nn(Xrot, dx=1, orig=X.min() - 0.5)
        # crop projections outside the detector
        mask = np.logical_and(inds >= 0, inds < l_x)
        detector_inds.append((inds[mask] + i * l_x).astype(np.int32))
        data_inds.append(data_unravel_indices[mask])
    detector_inds = np.concatenate(detector_inds)
    data_inds = np.concatenate(data_inds)
    weights = np.ones(len(data_inds), dtype=np.uint16)
    proj_operator = sparse.coo_matrix((weights, (detector_inds, data_inds)))
    return sparse.csr_matrix(proj_operator)


def _generate_center_coordinates(l_x):
    """
    Compute the coordinates of pixels centers for an image of
    linear size l_x
    """
    l_x = float(l_x)
    X, Y = np.mgrid[:l_x, :l_x]
    center = l_x / 2.
    X += 0.5 - center
    Y += 0.5 - center
    return X, Y

def _weights_nn(x, dx=1, orig=0, ravel=True):
    """
    Nearest-neighbour interpolation
    """
    if ravel:
        x = np.ravel(x)
    floor_x = np.floor(x - orig)
    return floor_x.astype(np.uint16)


