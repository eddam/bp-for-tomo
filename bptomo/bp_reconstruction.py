"""
Belief propagation iterations for binary tomography reconstruction
"""
import numpy as np
from solve_ising import solve_line
try:
    from joblib import Parallel, delayed
    joblib_import = True
except ImportError:
    joblib_import = False
if not joblib_import:
    try:
        from sklearn.externals.joblib import Parallel, delayed
        joblib_import = True
    except ImportError:
        joblib_import = False


def _reorder(inds, l_x):
    """
    Order inds in the order of a light ray of angle > pi/2, that is,
    with increasing x (first coordinate) and decreasing y (second coordinate)
    """
    ind_x, ind_y = inds / l_x, inds % l_x
    ind_y = l_x - ind_y
    return inds[np.argsort(ind_x * l_x + ind_y)]


def _calc_Jeff(inds, l_x, J):
    """
    Coupling between two neighbour spins.

    For a slanted light ray, the distance between spins is not 1. We
    therefore correct the coupling factor.

    Parameters
    ----------

    inds: tuple (i, j)
        positions of the two spins

    l_x: int
        size of the original image

    J: float
        coupling value for a distance of 1
    """
    x, y = inds / l_x, inds % l_x
    dist = np.abs(x[1:] - x[:-1]) + np.abs(y[1:] - y[:-1])
    res = np.tanh(J) ** dist
    res = .5 * np.log1p(res) - .5 * np.log1p(-res)
    return res


def _initialize_field(y, proj_operator, big_field=400):
    """
    Message passing from measurements to pixels, in the case where
    there is no spatial coupling between spins.

    Without coupling, the value of the field h_m_to_px (from measure to
    pixel)is given by

    h_m_to_px = atanh(y_m / N_m)

    where y_m is the measurement corresponding to light-ray, and N_m the
    number of spins contributing to this measurement.

    Parameters
    ----------

    y: ndarray
        measurements

    proj_operator: sparse matrix
        the tomography projection operator

    big_field:  float, optional
        value used to clip the field, to avoid numerical underflow
        and overflow

    Returns
    -------
    h_m_to_px
    """
    l_x = np.sqrt(proj_operator.shape[1])
    h_m_to_px = np.zeros((len(y) / l_x, l_x ** 2))
    for i, proj_value in enumerate(y):
        inds = proj_operator.rows[i]
        mu = i / int(l_x)
        ratio = proj_value / float(len(inds))
        if np.abs(ratio) >= 1:
            h_m_to_px[mu][inds] = big_field * np.sign(ratio)
        else:
            h_m_to_px[mu][inds] = 0.5 * (np.log1p(ratio) - \
                               np.log1p(-ratio))
    mask = np.abs(h_m_to_px) > big_field / 2.
    h_m_to_px[mask] = np.sign(h_m_to_px[mask]) * big_field / 2.
    return h_m_to_px


def _calc_hatf(h_m_to_px):
    """
    Computation of the field from pixels to measures.

    h_px_to_m[mu] = sum_nu h_m_to_px[nu]

    where the sum over nu is over all measurements to which a pixel
    contributes, except mu.
    """
    h_px_to_m = np.empty_like(h_m_to_px)
    h_sum = h_m_to_px.sum(axis=0)
    for mu_to_px, px_to_mu in zip(h_m_to_px, h_px_to_m):
        px_to_mu[:] = h_sum - mu_to_px
    return h_px_to_m, h_sum


def _calc_hatf_mf_correct(h_m_to_px):
    """
    Computation of the field from pixels to measures.

    h_px_to_m[mu] = sum_nu h_m_to_px[nu]

    where the sum over nu is over all measurements to which a pixel
    contributes, except mu.
    """
    l = len(h_m_to_px)
    h_sum = (1 - 1./l) * h_m_to_px.sum(axis=0)
    return h_sum



def _calc_hatf_mf(h_m_to_px):
    """
    Computation of the field from pixels to measures.

    h_px_to_m[mu] = sum_nu h_m_to_px[nu]

    where the sum over nu is over all measurements to which a pixel
    contributes, except mu.
    """
    h_sum = h_m_to_px.sum(axis=0)
    return h_sum


def BP_step_mf(h_m_to_px, H_px, y, proj_operator, J=.1,
                        use_mask=True, hext=None, mf_correct=True):
    """
    One iteration of BP (belief propagation), with messages updated
    after all new messages have been computed. A strong damping is needed
    with this update method.

    Parameters
    ----------

    h_m_to_px: ndarray of shape (ndir, l_x, l_x) where the original image
        is (l_x, l_x)
        local magnetizations (measure to field). The frame is the one of the
        non-rotated image (ie, Ising chains are taken at an angle
        corresponding to the projection)

    H_px: ndarray of shape (l_x, l_x) where the original image
        is (l_x, l_x)
        local magnetic field, determined by the other measures. The frame
        is the one of the non-rotated image (ie, Ising chains are taken at
        an angle corresponding to the projection)

    y: ndarray of shape ndir*L (1-D)
        Array that contains all measures (sum of spins along rays)

    proj_operator: sparse matrix, in lil format
        The projection operator (with only 1s in non-zero entries)

    J: float, default 0.1
        Amplitude of the coupling between spins. The larger J, the higher the
        coupling.

    use_mask: bool
        If True, there are no spins outside the central circle

    hext: ndarray of same shape as y, default None
        Guesses for the values of the external field to be used in the Ising
        chains.

    Returns
    -------

    h_m_to_px, h_px_to_m, h_sum, h_ext

    h_m_to_px and h_px_to_m are the new iterates

    h_sum is the sum over all directions of h_m_to_px, from which the marginal
    of a spin to be +1 or -1 can be computed.

    h_ext is the new external field
    """
    ndir = len(h_m_to_px)
    if hext is None:
        hext = np.zeros_like(y)
    hext_new = np.zeros_like(y)
    # Heuristic value that works well for the damping factor
    # The more projections, the more damping we need
    damping = 1 - 1.6 / ndir
    l_x = np.sqrt(h_m_to_px.shape[1])
    h_tmp = np.copy(h_m_to_px)
    if use_mask:
        X, Y = np.ogrid[:l_x, :l_x]
        mask = ((X - l_x / 2) ** 2 + (Y - l_x / 2) ** 2 <= \
                                    (l_x / 2) ** 2).ravel()
    for i, proj_value in enumerate(y):
        # Which pixels are in this measurement?
        inds = np.array(proj_operator.rows[i])
        # How are they ordered?
        if i > len(y) / 2:
            inds = _reorder(inds, l_x)
        # Handle pixels outside circle -- remove them
        mask_inds = mask[inds]
        inds = inds[mask_inds]
        if len(inds) == 0:
            continue
        # effective couplings
        Js = _calc_Jeff(inds, l_x, J)
        mu = i / int(l_x)  # angle number
        # Solve the chain
        h_m_to_px[mu][inds], hext_new[i] = solve_line(H_px[inds], Js,
                        proj_value, hext=hext[i])
    h_m_to_px = (1 - damping) * h_m_to_px + damping * h_tmp
    # Then we update h_px_to_m
    if mf_correct:
        h_sum = _calc_hatf_mf_correct(h_m_to_px)
    else:
        h_sum = _calc_hatf_mf(h_m_to_px)
    h_sum[~mask] = 0
    return h_m_to_px, h_sum, hext_new


def BP_step(h_m_to_px, h_px_to_m, y, proj_operator, J=.1,
                        use_mask=True, hext=None):
    """
    One iteration of BP (belief propagation), with messages updated
    after all new messages have been computed. A strong damping is needed
    with this update method.

    Parameters
    ----------

    h_m_to_px: ndarray of shape (ndir, l_x, l_x) where the original image
        is (l_x, l_x)
        local magnetizations (measure to field). The frame is the one of the
        non-rotated image (ie, Ising chains are taken at an angle
        corresponding to the projection)

    h_px_to_m: ndarray of shape (ndir, l_x, l_x) where the original image
        is (l_x, l_x)
        local magnetic field, determined by the other measures. The frame
        is the one of the non-rotated image (ie, Ising chains are taken at
        an angle corresponding to the projection)

    y: ndarray of shape ndir*L (1-D)
        Array that contains all measures (sum of spins along rays)

    proj_operator: sparse matrix, in lil format
        The projection operator (with only 1s in non-zero entries)

    J: float, default 0.1
        Amplitude of the coupling between spins. The larger J, the higher the
        coupling.

    use_mask: bool
        If True, there are no spins outside the central circle

    hext: ndarray of same shape as y, default None
        Guesses for the values of the external field to be used in the Ising
        chains.

    Returns
    -------

    h_m_to_px, h_px_to_m, h_sum, h_ext

    h_m_to_px and h_px_to_m are the new iterates

    h_sum is the sum over all directions of h_m_to_px, from which the marginal
    of a spin to be +1 or -1 can be computed.

    h_ext is the new external field
    """
    ndir = len(h_m_to_px)
    if hext is None:
        hext = np.zeros_like(y)
    hext_new = np.zeros_like(y)
    # Heuristic value that works well for the damping factor
    # The more projections, the more damping we need
    damping = 1 - 1.6 / ndir
    l_x = np.sqrt(h_m_to_px.shape[1])
    h_tmp = np.copy(h_m_to_px)
    if use_mask:
        X, Y = np.ogrid[:l_x, :l_x]
        mask = ((X - l_x / 2) ** 2 + (Y - l_x / 2) ** 2 <= \
                                    (l_x / 2) ** 2).ravel()
    for i, proj_value in enumerate(y):
        # Which pixels are in this measurement?
        inds = np.array(proj_operator.rows[i])
        # How are they ordered?
        if i > len(y) / 2:
            inds = _reorder(inds, l_x)
        # Handle pixels outside circle -- remove them
        mask_inds = mask[inds]
        inds = inds[mask_inds]
        if len(inds) == 0:
            continue
        # effective couplings
        Js = _calc_Jeff(inds, l_x, J)
        mu = i / int(l_x)  # angle number
        # Solve the chain
        h_m_to_px[mu][inds], hext_new[i] = solve_line(h_px_to_m[mu][inds], Js,
                        proj_value, hext=hext[i])
    h_m_to_px = (1 - damping) * h_m_to_px + damping * h_tmp
    # Then we update h_px_to_m
    h_px_to_m, h_sum = _calc_hatf(h_m_to_px)
    h_sum[~mask] = 0
    return h_m_to_px, h_px_to_m, h_sum, hext_new


def BP_step_parallel(h_m_to_px, h_px_to_m, y, proj_operator, J=.1,
                        use_mask=True, hext=None):
    """
    One iteration of BP (belief propagation), with messages updated
    after all new messages have been computed. A strong damping is needed
    with this update method. This function is parallelized (uses as many
    procs as possible !)

    Parameters
    ----------

    h_m_to_px: ndarray of shape (ndir, l_x, l_x) where the original image
        is (l_x, l_x)
        local magnetizations (measure to field). The frame is the one of the
        non-rotated image (ie, Ising chains are taken at an angle
        corresponding to the projection)

    h_px_to_m: ndarray of shape (ndir, l_x, l_x) where the original image
        is (l_x, l_x)
        local magnetic field, determined by the other measures. The frame
        is the one of the non-rotated image (ie, Ising chains are taken at
        an angle corresponding to the projection)

    y: ndarray of shape ndir*L (1-D)
        Array that contains all measures (sum of spins along rays)

    proj_operator: sparse matrix, in lil format
        The projection operator (with only 1s in non-zero entries)

    J: float
        Amplitude of the coupling between spins

    use_mask: bool
         If True, there are no spins outside the central circle

    hext: ndarray of same shape as y, default None
        Guesses for the values of the external field to be used in the Ising
        chains.


    Returns
    -------

    h_m_to_px, h_px_to_m, h_sum

    h_m_to_px and h_px_to_m are the new iterates

    h_sum is the sum over all directions of h_m_to_px, from which the marginal
    of a spin to be +1 or -1 can be computed.

    h_ext is the new external field
    """
    if not joblib_import:
        raise ImportError("""joblib could not be imported, use the
        function BP_step instead""")
    ndir = len(h_m_to_px)
    if hext is None:
        hext = np.zeros_like(y)
    hext_new = np.zeros_like(y)
    # Heuristic value that works well for the damping factor
    # The more projections, the more damping we need
    damping = 1 - 1.6 / ndir
    l_x = np.sqrt(h_m_to_px.shape[1])
    h_tmp = np.copy(h_m_to_px)
    if use_mask:
        X, Y = np.ogrid[:l_x, :l_x]
        mask = ((X - l_x / 2) ** 2 + (Y - l_x / 2) ** 2 <= \
                        (l_x / 2) ** 2).ravel()
    inds_all = []
    J_all = []
    # We should not recompute this every time, but pass to the function
    # the indices of the chains, and the effective coupling
    for i, proj_value in enumerate(y):
        # Which pixels are in this measurement?
        inds = np.array(proj_operator.rows[i])
        # How are they ordered?
        if i > len(y) / 2:
            inds = _reorder(inds, l_x)
        # Handle pixels outside circle -- remove them
        mask_inds = mask[inds]
        inds = inds[mask_inds]
        inds_all.append(inds)
        if len(inds) == 0:
            continue
        # effective couplings
        Js = _calc_Jeff(inds, l_x, J)
        J_all.append(Js)
        mu = i / int(l_x)  # angle number
    # Solve the chain in parallel for all measurements
    res = Parallel(n_jobs=-1, verbose=0)(delayed(solve_line)(h_px_to_m[i/int(l_x)][inds], Js, proj_value, hext=hext_val) for i, (inds, Js, proj_value, hext_val) in enumerate(zip(inds_all, J_all, y, hext)))
    for i, (inds, resi) in enumerate(zip(inds_all, res)):
        h_m_to_px[i/int(l_x)][inds] = resi[0]
        hext_new[i] = resi[1]
    h_m_to_px = (1 - damping) * h_m_to_px + damping * h_tmp
    # Then we update h_px_to_m
    h_px_to_m, h_sum = _calc_hatf(h_m_to_px)
    h_sum[~mask] = 0
    return h_m_to_px, h_px_to_m, h_sum, hext_new
