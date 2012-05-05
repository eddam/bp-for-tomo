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
    l_x = np.sqrt(proj_operator.shape[1])
    h_m_to_px = np.zeros((len(y)/l_x, l_x**2))
    for i, proj_value in enumerate(y):
        inds = proj_operator[i].indices
        mu = i / int(l_x)
        ratio = proj_value / float(len(inds))
        if np.abs(ratio) == 1:
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

def BP_step(h_m_to_px, h_px_to_m, y, proj_operator, J=.1, damping=0.8):
    ndir = len(h_m_to_px)
    l_x = np.sqrt(h_m_to_px.shape[1])
    # First we update h_m_to_px, by solving the Ising chain
    # the pixels are rotated so that the measure is horizontal
    h_tmp = np.copy(h_m_to_px)
    for i, proj_value in enumerate(y):
        inds = proj_operator[i].indices
        if i > len(y)/2:
            inds = _reorder(inds, l_x)
        Js = _calc_Jeff(inds, l_x, J)
        mu = i / int(l_x)
        h_m_to_px[mu][inds] = solve_line(h_px_to_m[mu][inds], Js,
                        proj_value)
    h_m_to_px = (1 - damping) * h_m_to_px + damping * h_tmp
    # Then we update h_px_to_m
    h_px_to_m, h_sum = _calc_hatf(h_m_to_px)
    return h_m_to_px, h_px_to_m, h_sum

