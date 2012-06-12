import numpy as np
from solve_ising import solve_line

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


def _initialize_field(y, proj_operator, big_field=10):
    """
    Message passing from measurements to pixels, in the case where
    there is no spatial coupling between spins.
    """
    #TODO: use mask
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

def BP_step(h_m_to_px, h_px_to_m, y, proj_operator, J=.1, damping=0.8,
                        use_mask=True, use_micro=False):
    ndir = len(h_m_to_px)
    l_x = np.sqrt(h_m_to_px.shape[1])
    # First we update h_m_to_px, by solving the Ising chain
    # the pixels are rotated so that the measure is horizontal
    h_tmp = np.copy(h_m_to_px)
    if use_mask:
        X, Y = np.ogrid[:l_x, :l_x]
        mask = ((X - l_x/2)**2 + (Y - l_x/2)**2 <= (l_x/2)**2).ravel()
    for i, proj_value in enumerate(y):
        inds = np.array(proj_operator.rows[i])
        if i > len(y)/2:
            inds = _reorder(inds, l_x)
        mask_inds = mask[inds]
        inds = inds[mask_inds]
        if len(inds) == 0:
            continue
        Js = _calc_Jeff(inds, l_x, J)
        mu = i / int(l_x)
        h_m_to_px[mu][inds] = solve_line(h_px_to_m[mu][inds], Js,
                        proj_value, use_micro=use_micro)
    h_m_to_px = (1 - damping) * h_m_to_px + damping * h_tmp
    # Then we update h_px_to_m
    h_px_to_m, h_sum = _calc_hatf(h_m_to_px)
    h_sum[~mask] = 0
    return h_m_to_px, h_px_to_m, h_sum

def BP_step_always_update(h_m_to_px, h_px_to_m, y, proj_operator, 
                    J=.1, damping=0.8, use_mask=True, use_micro=False):
    ndir = len(h_m_to_px)
    l_x = np.sqrt(h_m_to_px.shape[1])
    # First we update h_m_to_px, by solving the Ising chain
    # the pixels are rotated so that the measure is horizontal
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
        h_m_to_px_line = solve_line(h_px_to_m[mu][inds], Js,
                        proj_value, use_micro=use_micro)
        damping = 0.2
        h_m_to_px_line = (1 - damping) * h_m_to_px_line \
                            + damping * h_tmp[mu][inds]
        h_m_to_px[mu][inds] = h_m_to_px_line[:]
        delta_h = h_m_to_px_line - h_tmp[mu][inds]
        h_px_to_m = _calc_hatf_partial(h_px_to_m, delta_h, mu, inds)
    # Then we update h_px_to_m
    h_sum = h_m_to_px.sum(axis=0)
    h_sum[~mask] = 0
    return h_m_to_px, h_px_to_m, h_sum

def _order_dir(ndir, batch_nb=8):
    """
    Change the order of directions so that they are spaced enough
    """
    dirs = range(ndir)
    step = ndir / batch_nb
    alldirs = [dirs[i::step] for i in range(step)]
    return sum(alldirs, [])

def BP_step_update_direction(h_m_to_px, h_px_to_m, y, proj_operator,
                    J=.1, damping=0.8, use_mask=True, use_micro=False):
    ndir = len(h_m_to_px)
    l_x = np.sqrt(h_m_to_px.shape[1])
    # First we update h_m_to_px, by solving the Ising chain
    # the pixels are rotated so that the measure is horizontal
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
            h_dir_tmp[inds] = solve_line(h_px_to_m[direction][inds], Js,
                        proj_value, use_micro=use_micro)
        damping = 0.1
        h_dir_tmp = (1 - damping) * h_dir_tmp + damping * h_tmp[direction]
        h_m_to_px[direction] = h_dir_tmp[:]
        delta_h = h_dir_tmp - h_tmp[direction]
        h_px_to_m = _calc_hatf_partial_dir(h_px_to_m, delta_h, direction)
    # Then we update h_px_to_m
    h_sum = h_m_to_px.sum(axis=0)
    h_sum[~mask] = 0
    return h_m_to_px, h_px_to_m, h_sum

