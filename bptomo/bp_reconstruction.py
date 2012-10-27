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
    Coupling between two indices
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
    """
    l_x = np.sqrt(proj_operator.shape[1])
    h_m_to_px = np.zeros((len(y)/l_x, l_x**2))
    for i, proj_value in enumerate(y):
        inds = proj_operator.rows[i]
        mu = i / int(l_x)
        ratio = proj_value / float(len(inds))
        if np.abs(ratio) >= 1:
            h_m_to_px[mu][inds] = big_field * np.sign(ratio)
        else:
            h_m_to_px[mu][inds] = 0.5 * (np.log1p(ratio) - \
                               np.log1p(-ratio))
    mask = np.abs(h_m_to_px) > big_field/2.
    h_m_to_px[mask] = np.sign(h_m_to_px[mask]) * big_field / 2.
    return h_m_to_px

def _calc_hatf(h_m_to_px):
    h_px_to_m = np.empty_like(h_m_to_px)
    h_sum = h_m_to_px.sum(axis=0)
    for mu_to_px, px_to_mu in zip(h_m_to_px, h_px_to_m):
        px_to_mu[:] = h_sum - mu_to_px
    return h_px_to_m, h_sum

def _calc_hatf_partial(h_px_to_m, delta_h, mu, inds):
    tmp = np.copy(h_px_to_m[mu])
    h_px_to_m[:, inds] += delta_h[:]
    h_px_to_m[mu] = tmp[:]
    return h_px_to_m

def _calc_hatf_partial_dir(h_px_to_m, delta_h, mu):
    tmp = np.copy(h_px_to_m[mu])
    h_px_to_m += delta_h[:]
    h_px_to_m[mu] = tmp[:]
    return h_px_to_m

def BP_step(h_m_to_px, h_px_to_m, y, proj_operator, J=.1,
                        use_mask=True, use_micro=False, hext=None):
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

    J: float
        Amplitude of the coupling between spins

    use_mask: bool
        If True, there are no spins outside the central circle

    use_micro: bool
        If True, a microcanonical computation is performed for short chains,
        or chains with few non-blocked spins. If False, only canonical
        computations are performed.

    Returns
    -------

    h_m_to_px, h_px_to_m, h_sum

    h_m_to_px and h_px_to_m are the new iterates

    h_sum is the sum over all directions of h_m_to_px, from which the marginal
    of a spin to be +1 or -1 can be computed.
    """
    ndir = len(h_m_to_px)
    if hext is None:
        hext = np.zeros_like(y)
    hext_new = np.zeros_like(y)
    # Heuristic value that works well for the damping factor
    # The more projections, the more damping we need
    damping = 1 - 1.6/ndir
    l_x = np.sqrt(h_m_to_px.shape[1])
    h_tmp = np.copy(h_m_to_px)
    if use_mask:
        X, Y = np.ogrid[:l_x, :l_x]
        mask = ((X - l_x/2)**2 + (Y - l_x/2)**2 <= (l_x/2)**2).ravel()
    for i, proj_value in enumerate(y):
        # Which pixels are in this measurement?
        inds = np.array(proj_operator.rows[i])
        # How are they ordered?
        if i > len(y)/2:
            inds = _reorder(inds, l_x)
        # Handle pixels outside circle -- remove them
        mask_inds = mask[inds]
        inds = inds[mask_inds]
        if len(inds) == 0:
            continue
        # effective couplings
        Js = _calc_Jeff(inds, l_x, J)
        mu = i / int(l_x) # angle number
        # Solve the chain
        h_m_to_px[mu][inds], hext_new[i] = solve_line(h_px_to_m[mu][inds], Js,
                        proj_value, hext=hext[i], use_micro=use_micro)
    h_m_to_px = (1 - damping) * h_m_to_px + damping * h_tmp
    # Then we update h_px_to_m
    h_px_to_m, h_sum = _calc_hatf(h_m_to_px)
    h_sum[~mask] = 0
    return h_m_to_px, h_px_to_m, h_sum, hext_new


def BP_step_parallel(h_m_to_px, h_px_to_m, y, proj_operator, J=.1,
                        use_mask=True, use_micro=False, hext=None):
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

    use_micro: bool
        If True, a microcanonical computation is performed for short chains,
        or chains with few non-blocked spins. If False, only canonical
        computations are performed.

    Returns
    -------

    h_m_to_px, h_px_to_m, h_sum

    h_m_to_px and h_px_to_m are the new iterates

    h_sum is the sum over all directions of h_m_to_px, from which the marginal
    of a spin to be +1 or -1 can be computed.
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
    damping = 1 - 1.6/ndir
    l_x = np.sqrt(h_m_to_px.shape[1])
    h_tmp = np.copy(h_m_to_px)
    if use_mask:
        X, Y = np.ogrid[:l_x, :l_x]
        mask = ((X - l_x/2)**2 + (Y - l_x/2)**2 <= (l_x/2)**2).ravel()
    inds_all = []
    J_all = []
    # We should not recompute this every time, but pass to the function
    # the indices of the chains, and the effective coupling
    for i, proj_value in enumerate(y):
        # Which pixels are in this measurement?
        inds = np.array(proj_operator.rows[i])
        # How are they ordered?
        if i > len(y)/2:
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
        mu = i / int(l_x) # angle number
    # Solve the chain
    res = Parallel(n_jobs=-1, verbose=0)(delayed(solve_line)(h_px_to_m[i/int(l_x)][inds], Js, proj_value, use_micro=use_micro, hext=hext_val) for i, (inds, Js, proj_value, hext_val) in enumerate(zip(inds_all, J_all, y, hext)))
    for i, (inds, resi) in enumerate(zip(inds_all, res)):
        h_m_to_px[i/int(l_x)][inds] = resi[0]
        hext_new[i] = resi[1]
    h_m_to_px = (1 - damping) * h_m_to_px + damping * h_tmp
    # Then we update h_px_to_m
    h_px_to_m, h_sum = _calc_hatf(h_m_to_px)
    h_sum[~mask] = 0
    return h_m_to_px, h_px_to_m, h_sum, hext_new

def BP_step_always_update(h_m_to_px, h_px_to_m, y, proj_operator,
                    J=.1, use_mask=True, use_micro=False, hext=None):
    """
    One iteration of BP (belief propagation), with messages updated
    after each resolution of an Ising chain.

    A new random permutation of the chains is computed to determine the
    order in which the chains are processed.

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

    use_micro: bool
        If True, a microcanonical computation is performed for short chains,
        or chains with few non-blocked spins. If False, only canonical
        computations are performed.

    Returns
    -------

    h_m_to_px, h_px_to_m, h_sum

    h_m_to_px and h_px_to_m are the new iterates

    h_sum is the sum over all directions of h_m_to_px, from which the marginal
    of a spin to be +1 or -1 can be computed.
    """

    ndir = len(h_m_to_px)
    if hext is None:
        hext = np.zeros_like(y)
    hext_new = np.zeros_like(y)
    l_x = np.sqrt(h_m_to_px.shape[1]) 
    h_tmp = np.copy(h_m_to_px)
    if use_mask:
        X, Y = np.ogrid[:l_x, :l_x]
        mask = ((X - l_x/2)**2 + (Y - l_x/2)**2 <= (l_x/2)**2).ravel()
    order = np.random.permutation(range(len(y)))
    for measure_index in order:
        proj_value = y[measure_index]
        inds = np.array(proj_operator.rows[measure_index])
        if measure_index > len(y)/2:
            inds = _reorder(inds, l_x)
        mask_inds = mask[inds]
        inds = inds[mask_inds]
        if len(inds) == 0:
            continue
        Js = _calc_Jeff(inds, l_x, J)
        mu = measure_index / int(l_x)
        h_m_to_px_line, hext_new[measure_index] = \
                solve_line(h_px_to_m[mu][inds], Js,
                    proj_value, use_micro=use_micro, hext=hext[measure_index])
        damping = 0.3#5
        h_m_to_px_line = (1 - damping) * h_m_to_px_line \
                            + damping * h_tmp[mu][inds]
        h_m_to_px[mu][inds] = h_m_to_px_line[:]
        delta_h = h_m_to_px_line - h_tmp[mu][inds]
        h_px_to_m = _calc_hatf_partial(h_px_to_m, delta_h, mu, inds)
    # Then we update h_px_to_m
    h_sum = h_m_to_px.sum(axis=0)
    h_sum[~mask] = 0
    return h_m_to_px, h_px_to_m, h_sum, hext_new

def _order_dir(ndir, batch_nb=8):
    """
    Change the order of directions so that they are spaced enough;
    If there are ndir directions, they are processed in the order
    i + j * ndir / 8 for j in range(0, 8), for i in range(0, ndir/8)
    """
    dirs = range(ndir)
    step = ndir / batch_nb
    alldirs = [dirs[i::step] for i in range(step)]
    return sum(alldirs, [])

def BP_step_update_direction(h_m_to_px, h_px_to_m, y, proj_operator,
                    J=.1, use_mask=True, use_micro=False, hext=None):
    """
    One iteration of BP (belief propagation), with messages updated
    after all messages corresponding to one angle have been computed.

    The directions are processed so that closeby directions are not processed
    together. If there are ndir directions, they are processed in the order
    i + j * ndir / 8 for j in range(0, 8), for i in range(0, ndir/8)

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

    use_micro: bool
        If True, a microcanonical computation is performed for short chains,
        or chains with few non-blocked spins. If False, only canonical
        computations are performed.

    hext : ndarray of same shape as y
        Initial guess for the external fields.

    Returns
    -------

    h_m_to_px, h_px_to_m, h_sum

    h_m_to_px and h_px_to_m are the new iterates

    h_sum is the sum over all directions of h_m_to_px, from which the marginal
    of a spin to be +1 or -1 can be computed.

    Notes
    -----
    The steps performed here are as follows:

    1. Loop over the projection angles
    2. Loop over measures for a given projection angle
    3. Compute effective couplings Js
    4. Solve the line

    Both types of fields (h_px_to_m and h_m_to_px) are in the image frame,
    and indices corresponding to the different light-rays are extracted
    from the projection operator.

    """

    ndir = len(h_m_to_px)
    if hext is None:
        hext = np.zeros_like(y)
    hext_new = np.zeros_like(y)
    l_x = np.sqrt(h_m_to_px.shape[1])
    h_tmp = np.copy(h_m_to_px)
    if use_mask:
        X, Y = np.ogrid[:l_x, :l_x]
        mask = ((X - l_x/2)**2 + (Y - l_x/2)**2 <= (l_x/2)**2).ravel()
    order = _order_dir(ndir)
    for direction in order:
        h_dir_tmp = np.zeros_like(h_px_to_m[direction])
        for pixel_ind in range(int(l_x)):
            measure_index = l_x * direction + pixel_ind
            proj_value = y[measure_index]
            inds = np.array(proj_operator.rows[measure_index])
            if measure_index > len(y)/2:
                inds = _reorder(inds, l_x)
            mask_inds = mask[inds]
            inds = inds[mask_inds]
            if len(inds) == 0:
                continue
            Js = _calc_Jeff(inds, l_x, J)
            h_dir_tmp[inds], hext_new[measure_index] = \
                solve_line(h_px_to_m[direction][inds], Js,
                    proj_value, use_micro=use_micro, hext=hext[measure_index])
        damping = 0.9
        h_old = np.copy(h_m_to_px[direction])
        h_dir_tmp = (1 - damping) * h_dir_tmp + damping * h_old
        h_m_to_px[direction] = h_dir_tmp[:]
        delta_h = h_dir_tmp - h_old
        h_px_to_m = _calc_hatf_partial_dir(h_px_to_m, delta_h, direction)
    h_sum = h_m_to_px.sum(axis=0)
    h_sum[~mask] = 0
    return h_m_to_px, h_px_to_m, h_sum, hext_new

def BP_parallel_update_direction(h_m_to_px, h_px_to_m, y, proj_operator,
                    J=.1, use_mask=True, use_micro=False, hext=None):
    """
    One iteration of BP (belief propagation), with messages updated
    after all messages corresponding to one angle have been computed.

    The directions are processed so that closeby directions are not processed
    together. If there are ndir directions, they are processed in the order
    i + j * ndir / 8 for j in range(0, 8), for i in range(0, ndir/8)

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

    use_micro: bool
        If True, a microcanonical computation is performed for short chains,
        or chains with few non-blocked spins. If False, only canonical
        computations are performed.

    hext : ndarray of same shape as y
        Initial guess for the external fields.

    Returns
    -------

    h_m_to_px, h_px_to_m, h_sum

    h_m_to_px and h_px_to_m are the new iterates

    h_sum is the sum over all directions of h_m_to_px, from which the marginal
    of a spin to be +1 or -1 can be computed.

    Notes
    -----
    The steps performed here are as follows:

    1. Loop over the projection angles
    2. Loop over measures for a given projection angle
    3. Compute effective couplings Js
    4. Solve the line

    Both types of fields (h_px_to_m and h_m_to_px) are in the image frame,
    and indices corresponding to the different light-rays are extracted
    from the projection operator.

    """
    if not joblib_import:
        raise ImportError("""joblib could not be imported, use the
        function BP_step instead""")
    ndir = len(h_m_to_px)
    if hext is None:
        hext = np.zeros_like(y)
    hext_new = np.zeros_like(y)
    l_x = np.sqrt(h_m_to_px.shape[1])
    h_tmp = np.copy(h_m_to_px)
    if use_mask:
        X, Y = np.ogrid[:l_x, :l_x]
        mask = ((X - l_x/2)**2 + (Y - l_x/2)**2 <= (l_x/2)**2).ravel()
    order = _order_dir(ndir)
    for direction in order:
        h_dir_tmp = np.zeros_like(h_px_to_m[direction])
        for pixel_ind in range(int(l_x)):
            measure_index = l_x * direction + pixel_ind
            proj_value = y[measure_index]
            inds = np.array(proj_operator.rows[measure_index])
            if measure_index > len(y)/2:
                inds = _reorder(inds, l_x)
            mask_inds = mask[inds]
            inds = inds[mask_inds]
            if len(inds) == 0:
                continue
            Js = _calc_Jeff(inds, l_x, J)
            h_dir_tmp[inds], hext_new[measure_index] = \
                solve_line(h_px_to_m[direction][inds], Js,
                    proj_value, use_micro=use_micro, hext=hext[measure_index])
        damping = 0.9
        h_old = np.copy(h_m_to_px[direction])
        h_dir_tmp = (1 - damping) * h_dir_tmp + damping * h_old
        h_m_to_px[direction] = h_dir_tmp[:]
        delta_h = h_dir_tmp - h_old
        h_px_to_m = _calc_hatf_partial_dir(h_px_to_m, delta_h, direction)
    h_sum = h_m_to_px.sum(axis=0)
    h_sum[~mask] = 0
    return h_m_to_px, h_px_to_m, h_sum, hext_new


